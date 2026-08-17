"""
Prompt Routing Engine.

在线分类 + 模型映射，由 custom_auth.py PromptRouter 调用。
支持热加载 routing_rules.yaml 配置，策略闭环可自动更新路由规则。

三类分类器:
  1. heuristic — 关键词/长度/语言启发式（零延迟）
  2. llm       — LLM 分类（可配置模型，适合复杂场景）
  3. upstream  — 依赖上游服务/外部 API 分类
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger("pricing_agent.router")

# ── 数据模型 ────────────────────────────────────────────────────

ROUTING_CATEGORIES = ["simple", "complex", "chinese", "creative", "agentic", "default"]


@dataclass
class RoutingCondition:
    condition_type: str       # max_length / min_length / has_keyword / has_code / language / regex
    value: object = None
    keywords: list[str] = field(default_factory=list)
    pattern: str = ""


@dataclass
class CategoryRule:
    name: str
    model: str
    conditions: list[RoutingCondition] = field(default_factory=list)
    priority: int = 0
    candidates: list[str] = field(default_factory=list)  # 候选模型列表，非空时按成本最低选择


@dataclass
class RoutingConfig:
    default_model: str = "deepseek-v4-flash"
    categories: list[CategoryRule] = field(default_factory=list)
    llm_classifier_model: str = ""
    llm_classifier_prompt: str = ""


@dataclass
class RoutingDecision:
    category: str
    assigned_model: str
    confidence: float
    original_model: str = ""
    method: str = "heuristic"   # heuristic / llm / upstream / passthrough


# ── 配置加载 ────────────────────────────────────────────────────

_CONFIG_PATH: str = ""
_CONFIG_CACHE: RoutingConfig = RoutingConfig()
_CONFIG_MTIME: float = 0


def _default_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "routing_rules.yaml",
    )


def set_config_path(path: str):
    global _CONFIG_PATH
    _CONFIG_PATH = path


def load_config(force: bool = False) -> RoutingConfig:
    global _CONFIG_CACHE, _CONFIG_MTIME, _CONFIG_PATH
    if not _CONFIG_PATH:
        _CONFIG_PATH = os.environ.get("ROUTING_CONFIG_PATH", _default_config_path())
    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
        if not force and mtime <= _CONFIG_MTIME:
            return _CONFIG_CACHE
    except FileNotFoundError:
        logger.warning("Routing config not found: %s", _CONFIG_PATH)
        return _CONFIG_CACHE

    try:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to load routing config: %s", exc)
        return _CONFIG_CACHE

    routing_raw = raw.get("routing", {})
    cats = []
    for c in routing_raw.get("categories", []):
        conds = []
        for cd in c.get("conditions", []):
            conds.append(RoutingCondition(
                condition_type=cd.get("type", ""),
                value=cd.get("value"),
                keywords=cd.get("keywords", []),
                pattern=cd.get("pattern", ""),
            ))
        cats.append(CategoryRule(
            name=c.get("name", "default"),
            model=c.get("model", routing_raw.get("default_model", "deepseek-v4-flash")),
            conditions=conds,
            priority=c.get("priority", 0),
            candidates=c.get("candidates", []),
        ))

    _CONFIG_CACHE = RoutingConfig(
        default_model=routing_raw.get("default_model", "deepseek-v4-flash"),
        categories=sorted(cats, key=lambda x: -x.priority),
        llm_classifier_model=routing_raw.get("llm_classifier_model", ""),
        llm_classifier_prompt=routing_raw.get(
            "llm_classifier_prompt",
            "Classify this user message into one category: simple, complex, chinese, creative, agentic. Reply with just the category name.",
        ),
    )
    _CONFIG_MTIME = mtime
    logger.info("Loaded %d routing categories from %s", len(cats), _CONFIG_PATH)
    return _CONFIG_CACHE


def reload_config():
    return load_config(force=True)


# ── 启发式分类器 ────────────────────────────────────────────────

_CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CODE_KEYWORDS = [
    "def ", "class ", "import ", "function", "async ", "await ",
    "```", "const ", "let ", "var ", "#include", "impl ",
    "fn ", "pub ", "struct ", "trait ", "interface ", "extends ",
    "curl ", "http", "api_key", "endpoint", "docker", "kubectl",
    "npm ", "pip ", "cargo ", "git ", "mysql", "select ", "from ",
    "where ", "join ", "delete ", "insert ", "create table",
    "return ", "if ", "else:", "for ", "while ", "try:", "except",
    "lambda", "yield", "raise ", "assert ", "with ", "pass",
    "null", "undefined", "console.", ".map(", ".filter(", ".reduce(",
    "=>", "-> ", "::", "<?php", "</", "#!/",
    "int ", "float ", "str ", "bool ", "void ", "char ",
    "public ", "private ", "static ", "protected ", "package ",
    "systemd", "kubectl", "docker-compose", "Dockerfile",
]
_CREATIVE_KEYWORDS = [
    "poem", "story", "write a", "create a", "generate", "compose",
    "song", "歌词", "诗", "故事", "小说", "剧本", "essay",
    "tale", "narrative", "fiction", "creative writing",
]


def _contains_chinese(text: str) -> bool:
    return bool(_CHINESE_CHAR_PATTERN.search(text))


def _count_code_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in _CODE_KEYWORDS if kw.lower() in text_lower)


def _count_creative_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in _CREATIVE_KEYWORDS if kw.lower() in text_lower)


def _classify_heuristic(
    text: str,
    config: RoutingConfig,
) -> tuple[CategoryRule, float]:
    """启发式分类：返回 (matched_rule, confidence)。

    返回匹配到的 CategoryRule（含 model 信息），避免二次查找。
    """
    text_len = len(text)
    has_code = _count_code_keywords(text) >= 1
    has_chinese = _contains_chinese(text)
    has_creative = _count_creative_keywords(text) >= 1

    for cat in config.categories:
        matched = True
        match_count = 0
        total_conds = len(cat.conditions)
        if total_conds == 0:
            continue
        for cond in cat.conditions:
            cond_matched = True
            if cond.condition_type == "max_length":
                cond_matched = text_len <= (cond.value or 100)
            elif cond.condition_type == "min_length":
                cond_matched = text_len >= (cond.value or 500)
            elif cond.condition_type == "has_keyword":
                kw = cond.keywords or []
                cond_matched = any(k.lower() in text.lower() for k in kw)
            elif cond.condition_type == "has_code":
                cond_matched = has_code if cond.value is not False else not has_code
            elif cond.condition_type == "language":
                cond_matched = has_chinese if cond.value == "zh" else not has_chinese
            elif cond.condition_type == "has_creative":
                cond_matched = has_creative if cond.value is not False else not has_creative
            elif cond.condition_type == "regex":
                cond_matched = bool(cond.pattern and re.search(cond.pattern, text))
            if cond_matched:
                match_count += 1
            else:
                matched = False
        if matched:
            confidence = match_count / max(total_conds, 1)
            return cat, min(1.0, confidence + 0.3)

    # 兜底: 无规则匹配时走启发式，作为 default 类别返回
    fallback_model = config.default_model
    fallback_name = "default"
    if has_chinese:
        fallback_name = "chinese"
        fallback_model = "bedi/glm-4.7"
    elif text_len >= 80 and (has_code or text_len > 200):
        fallback_name = "complex"
        fallback_model = "deepseek-v4-pro"

    return CategoryRule(name=fallback_name, model=fallback_model), 0.5


def _classify_llm(text: str, config: RoutingConfig) -> tuple[CategoryRule, float]:
    """用 LLM 分类。失败时 fallback 到启发式。"""
    if not config.llm_classifier_model:
        return _classify_heuristic(text, config)
    try:
        import litellm
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        resp = litellm.completion(
            model=config.llm_classifier_model,
            api_key=api_key or None,
            messages=[
                {"role": "system", "content": config.llm_classifier_prompt},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in ROUTING_CATEGORIES:
            # 找到匹配的规则，用它的 model 映射
            for cat in config.categories:
                if cat.name == label:
                    return cat, 0.9
            return CategoryRule(name=label, model=config.default_model), 0.9
        logger.warning("LLM returned unknown category: %s, falling back", label)
    except Exception as exc:
        logger.warning("LLM classification failed: %s", exc)
    return _classify_heuristic(text, config)


# ── LLM 分类器 ──────────────────────────────────────────────────


def _classify_llm(text: str, config: RoutingConfig) -> tuple[str, float]:
    """用 LLM 分类。失败时 fallback 到启发式。"""
    if not config.llm_classifier_model:
        return _classify_heuristic(text, config)
    try:
        import litellm
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        resp = litellm.completion(
            model=config.llm_classifier_model,
            api_key=api_key or None,
            messages=[
                {"role": "system", "content": config.llm_classifier_prompt},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in ROUTING_CATEGORIES:
            return label, 0.9
        logger.warning("LLM returned unknown category: %s, falling back", label)
    except Exception as exc:
        logger.warning("LLM classification failed: %s", exc)
    return _classify_heuristic(text, config)


# ── 主路由入口 ──────────────────────────────────────────────────


def route(
    messages: list[dict],
    original_model: str = "",
    method: str = "heuristic",
) -> RoutingDecision:
    """对一组 messages 做路由决策。

    Args:
        messages: OpenAI 格式 message 列表
        original_model: 用户请求中指定的原始 model
        method: 分类方法 (heuristic / llm)

    Returns:
        RoutingDecision
    """
    config = load_config()

    # 提取最后一条 user 消息
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            user_msg = str(content)
            break

    if not user_msg:
        return RoutingDecision(
            category="default",
            assigned_model=config.default_model,
            confidence=1.0,
            original_model=original_model,
            method="passthrough",
        )

    if method == "llm":
        matched_rule, confidence = _classify_llm(user_msg, config)
    else:
        matched_rule, confidence = _classify_heuristic(user_msg, config)

    assigned = matched_rule.model
    if matched_rule.candidates:
        assigned = _pick_cheapest(matched_rule.candidates, matched_rule.model)

    return RoutingDecision(
        category=matched_rule.name,
        assigned_model=assigned,
        confidence=confidence,
        original_model=original_model,
        method=method,
    )


def _pick_cheapest(candidates: list, fallback: str) -> str:
    """从候选模型中选择综合成本（输入+输出均价）最低的模型。
    成本数据来自 proxy_server_config.yaml 中各模型的 input/output_cost_per_token。
    无成本数据的模型按无穷大处理；全部缺失时返回 fallback。
    """
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "proxy_server_config.yaml"
    )
    try:
        with open(cfg_path, encoding="utf-8") as f:
            proxy_cfg = yaml.safe_load(f)
    except Exception as e:
        logger.warning("cost-aware routing: failed to load proxy config: %s", e)
        return fallback
    costs = {}
    for entry in proxy_cfg.get("model_list", []):
        name = entry.get("model_name", "")
        params = entry.get("litellm_params", {})
        cin = params.get("input_cost_per_token")
        cout = params.get("output_cost_per_token")
        if cin is not None and cout is not None:
            costs[name] = (float(cin) + float(cout)) / 2.0
    best, best_cost = fallback, float("inf")
    for model in candidates:
        c = costs.get(model, float("inf"))
        if c < best_cost:
            best, best_cost = model, c
    if best != fallback:
        logger.info(
            "cost-aware routing: picked %s (avg_cost=%.3g) from %s",
            best, best_cost, candidates,
        )
    return best
