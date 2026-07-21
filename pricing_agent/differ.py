import os
from typing import Optional
from .models import ModelPrice, PriceChange, PriceSnapshot

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "proxy_server_config.yaml")


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _config_model_prices(
    config: dict,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for entry in config.get("model_list", []):
        lp = entry.get("litellm_params", {})
        model = lp.get("model", entry.get("model_name", ""))
        overrides = {}
        for field in (
            "input_cost_per_token",
            "output_cost_per_token",
            "cache_read_input_token_cost",
        ):
            val = lp.get(field)
            if val is not None:
                overrides[field] = float(val)
        result[model] = overrides
    return result


def _price_key(a: ModelPrice) -> str:
    return f"{a.provider}/{a.model_name}"


def diff_snapshot(
    snapshot: PriceSnapshot,
    config_path: Optional[str] = None,
) -> list[PriceChange]:
    path = config_path or CONFIG_PATH
    config = _load_yaml_config(path)
    return _diff(snapshot, config)


def _diff(snapshot: PriceSnapshot, config: dict) -> list[PriceChange]:
    changes: list[PriceChange] = []
    config_prices = _config_model_prices(config)
    model_entries = config.get("model_list", [])

    configured_models = {}
    for entry in model_entries:
        mn = entry.get("model_name", "")
        lp = entry.get("litellm_params", {})
        configured_models[mn] = lp.get("model", mn)

    price_lookup: dict[str, ModelPrice] = {}
    for mp in snapshot.models:
        key = _price_key(mp)
        price_lookup[mp.model_name] = mp
        price_lookup[key] = mp

    fetched_names = {m.model_name for m in snapshot.models}

    for model_name, actual_model_key in configured_models.items():
        mp = price_lookup.get(model_name) or price_lookup.get(actual_model_key)
        if mp is None:
            for k, v in price_lookup.items():
                if model_name in k or (actual_model_key and actual_model_key in k):
                    mp = v
                    break
        if mp is None:
            changes.append(PriceChange(
                change_type="config_mismatch",
                model_name=model_name,
                description=(
                    f"configured model '{model_name}' ({actual_model_key}) "
                    "not found in vendor price snapshot"
                ),
                suggested_action="verify model name, or ignore if deprecated",
                impact="spend tracking may be inaccurate without known prices",
            ))
            continue

        overrides = config_prices.get(actual_model_key or model_name, {})
        if overrides:
            continue

        changed = _check_model_prices(model_name, mp, config, changes)
        changes.extend(changed)

    for mp in snapshot.models:
        if mp.model_name not in configured_models and mp.model_name not in {
            v for v in configured_models.values()
        }:
            changes.append(PriceChange(
                change_type="model_added",
                model_name=mp.model_name,
                description=(
                    f"new model '{mp.model_name}' available at "
                    f"${mp.input_cost_per_token:.2e}/${mp.output_cost_per_token:.2e} "
                    "per token"
                ),
                suggested_action=(
                    "review and add to proxy_server_config.yaml model_list "
                    "if desired"
                ),
                impact="new capability available; check if needed for your use case",
            ))

    return changes


def _check_model_prices(
    model_name: str,
    fetched: ModelPrice,
    config: dict,
    existing: list,
) -> list[PriceChange]:
    changes: list[PriceChange] = []
    try:
        import litellm
        internal = litellm.model_cost.get(fetched.model_name) or litellm.model_cost.get(
            f"deepseek/{fetched.model_name}"
        )
    except Exception:
        internal = None

    refs = {}
    if internal:
        for fld in ("input_cost_per_token", "output_cost_per_token",
                     "cache_read_input_token_cost", "input_cost_per_token_cache_hit"):
            v = internal.get(fld)
            if v is not None:
                refs[fld] = float(v)

    fields_to_check = [
        ("input_cost_per_token", "input price"),
        ("output_cost_per_token", "output price"),
        ("cache_read_input_token_cost", "cache read price"),
    ]
    for fld, label in fields_to_check:
        fetched_val = getattr(fetched, fld, None)
        ref_val = refs.get(fld) or refs.get(
            {"cache_read_input_token_cost": "input_cost_per_token_cache_hit"}.get(fld, ""),
        )
        if fetched_val is None and ref_val is None:
            continue

        if ref_val is not None and fetched_val is not None:
            if abs(fetched_val - ref_val) / max(ref_val, 1e-12) > 0.01:
                changes.append(PriceChange(
                    change_type="price_change",
                    model_name=model_name,
                    field=fld,
                    old_value=ref_val,
                    new_value=fetched_val,
                    description=(
                        f"{label} changed: "
                        f"${ref_val:.2e} → ${fetched_val:.2e} per token "
                        f"({((fetched_val - ref_val) / ref_val) * 100:+.1f}%)"
                    ),
                    impact=_price_impact_text(fetched_val, ref_val, label),
                    suggested_action=(
                        "add price override in proxy_server_config.yaml "
                        f"if litellm's internal price is outdated. "
                        f"Use litellm_params.{fld}"
                    ),
                ))
        elif fetched_val is not None:
            changes.append(PriceChange(
                change_type="price_change",
                model_name=model_name,
                field=fld,
                new_value=fetched_val,
                description=f"{label}: ${fetched_val:.2e} (no previous reference)",
                suggested_action="verify this pricing is correct",
            ))

    if fetched.cache_read_input_token_cost is not None:
        old_cache = refs.get("input_cost_per_token_cache_hit")
        new_cache = fetched.cache_read_input_token_cost
        if old_cache is not None and new_cache is not None:
            ratio = new_cache / fetched.input_cost_per_token if fetched.input_cost_per_token else 0
            old_ratio = old_cache / refs.get("input_cost_per_token", 1) if refs.get("input_cost_per_token", 0) else 0
            if abs(ratio - old_ratio) > 0.01:
                changes.append(PriceChange(
                    change_type="caching_discount",
                    model_name=model_name,
                    field="cache_read_input_token_cost",
                    old_value=old_cache,
                    new_value=new_cache,
                    description=(
                        f"prompt caching discount changed: "
                        f"{old_ratio*100:.0f}% → {ratio*100:.0f}% of input price"
                    ),
                    impact="affects effective cost for repeated prompt prefixes",
                    suggested_action="update litellm_params.cache_read_input_token_cost",
                ))

    return changes


def _price_impact_text(new: float, old: float, label: str) -> str:
    pct = ((new - old) / old) * 100 if old else 0
    if pct < -5:
        return f"cost reduction: {label} down {abs(pct):.0f}% — "
        "good for users, check if budget forecasts need updating"
    elif pct > 5:
        return f"cost increase: {label} up {pct:.0f}% — "
        "may affect user budgets, consider adjusting USER_BUDGET_MAP"
    return f"minor change ({pct:+.1f}%), no action required"
