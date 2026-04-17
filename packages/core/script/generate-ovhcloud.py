#!/usr/bin/env python3
"""
Sync OVHcloud AI Endpoints models with providers/ovhcloud/models/ TOML files.
Source of truth: https://catalog.endpoints.ai.ovh.net/rest/v2/model

Usage:
  python3 generate-ovhcloud.py [--dry-run] [--verbose]
"""

import argparse
import re
import ssl
import tomllib
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

CATALOG_LIST   = "https://catalog.endpoints.ai.ovh.net/rest/v2/model"
CATALOG_DETAIL = "https://catalog.endpoints.ai.ovh.net/rest/v2/model/{id}"
OVHCLOUD_API   = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models"

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "providers" / "ovhcloud" / "models"

# Categories to include (LLM + Code + Visual + Guard)
INCLUDE_CATEGORIES = {
    "Large Language Models (LLM)",
    "Reasoning LLM",
    "Code LLM",
    "Visual LLM",
}


# --- Helpers -----------------------------------------------------------------

def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def parse_price(raw: str | float | int) -> float:
    """Parse '$0.09' or 0.09 → 0.09"""
    return float(str(raw).replace("$", "").strip())


def parse_context(raw: str | int) -> int:
    """Parse '131k' or 131000 → 131000"""
    if isinstance(raw, int):
        return raw
    s = str(raw).lower().strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(s)


def fmt_num(n: int) -> str:
    """131072 → 131_072"""
    s = str(n)
    if n < 1000:
        return s
    result = []
    for i, c in enumerate(reversed(s)):
        if i > 0 and i % 3 == 0:
            result.append("_")
        result.append(c)
    return "".join(reversed(result))


def fetch_json(url: str) -> dict | list:
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as r:
        return json.loads(r.read())


# --- Catalog parsing ---------------------------------------------------------

def alias_to_filename(aliases: list[str]) -> str:
    """
    Pick the best alias as TOML filename.
    Prefer the short alias without org prefix (e.g. 'llama-3.1-8b-instruct'),
    falling back to lowercased last part of org/model alias.
    """
    # Prefer aliases without '/' (short form)
    short = [a for a in aliases if "/" not in a]
    if short:
        return short[-1].lower() + ".toml"
    # Fall back: take part after last '/'
    return aliases[-1].split("/")[-1].lower() + ".toml"


def caps_from_catalog(detail: dict) -> dict:
    """Extract capability flags from catalog detail response."""
    caps = detail.get("capabilities") or detail.get("model_specs", {}).get("capabilities", {})
    response_formats = caps.get("response_format") or caps.get("response_formats") or []
    input_modality = caps.get("input_modality", ["text"])

    return {
        "reasoning":         bool(caps.get("reasoning", False)),
        "tool_call":         bool(caps.get("function_calling", False)),
        "structured_output": "json_schema" in response_formats,
        "attachment":        "image" in input_modality,
    }


def price_from_catalog(detail: dict) -> dict[str, float]:
    """Extract input/output prices from catalog detail."""
    pricing = detail.get("pricing") or detail.get("usage_information", {}).get("pricing")

    if isinstance(pricing, dict):
        # Format: {"input_tokens_per_million": "$0.09", "output_tokens_per_million": "$0.25"}
        inp = pricing.get("input_tokens_per_million") or pricing.get("input")
        out = pricing.get("output_tokens_per_million") or pricing.get("output")
        if inp is not None and out is not None:
            return {"input": parse_price(inp), "output": parse_price(out)}

    if isinstance(pricing, list):
        # Format: [{"price_unit": "million_input_tokens", "price": 0.67}, ...]
        inp = next((p["price"] for p in pricing if "input" in p.get("price_unit", "")), None)
        out = next((p["price"] for p in pricing if "output" in p.get("price_unit", "")), None)
        if inp is not None and out is not None:
            return {"input": float(inp), "output": float(out)}

    return {"input": 0.0, "output": 0.0}


def release_date_from_catalog(detail: dict) -> str | None:
    pub = detail.get("publishing_information", {})
    d = pub.get("release_date") or pub.get("created_at")
    if not d:
        return None
    # Try to normalize various date formats
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# --- /v1/models for context + output limits ----------------------------------

def build_v1_index(v1_models: list[dict]) -> dict[str, dict]:
    """Map normalized model ID → {context, output} limits."""
    index = {}
    for m in v1_models:
        key = m["id"].lower().replace("_", "-").replace(".", "-")
        index[key] = {
            "context": m.get("context_length", 0),
            "output":  m.get("max_completion_tokens", 0),
        }
    return index


def find_limits(catalog_id: str, aliases: list[str], v1_index: dict[str, dict]) -> dict | None:
    """Look up context + output limits from /v1/models."""
    candidates = [catalog_id] + [a.split("/")[-1] for a in aliases]
    for c in candidates:
        key = c.lower().replace("_", "-").replace(".", "-")
        if key in v1_index and v1_index[key]["context"] > 0:
            return v1_index[key]
    return None


# --- TOML I/O ----------------------------------------------------------------

def load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_model(detail: dict, caps: dict, cost: dict,
                limits: dict, existing: dict | None) -> dict:

    def keep(field, default):
        return existing.get(field, default) if existing else default

    release = release_date_from_catalog(detail) or keep("release_date", today())

    modalities_default = {
        "input": ["text", "image"] if caps["attachment"] else ["text"],
        "output": ["text"],
    }

    m = {
        "name":             detail.get("aliases", [detail["id"]])[0].split("/")[-1],
        "release_date":     keep("release_date", release),
        "last_updated":     today(),
        "attachment":       caps["attachment"],
        "reasoning":        caps["reasoning"],
        "tool_call":        caps["tool_call"],
        "structured_output": caps["structured_output"],
        "cost":             cost,
        "limit":            {"context": limits["context"], "output": limits["output"]},
        "modalities":       keep("modalities", modalities_default),
    }
    # Keep manual fields only if already set in the existing file
    for field in ("temperature", "open_weights"):
        if existing and field in existing:
            m[field] = existing[field]
    return m


def to_toml(m: dict) -> str:
    lines = []
    lines.append(f'name = "{m["name"]}"')
    lines.append(f'release_date = "{m["release_date"]}"')
    lines.append(f'last_updated = "{m["last_updated"]}"')
    lines.append(f'attachment = {str(m["attachment"]).lower()}')
    lines.append(f'reasoning = {str(m["reasoning"]).lower()}')
    lines.append(f'tool_call = {str(m["tool_call"]).lower()}')
    lines.append(f'structured_output = {str(m["structured_output"]).lower()}')
    if "temperature" in m:
        lines.append(f'temperature = {str(m["temperature"]).lower()}')
    if "open_weights" in m:
        lines.append(f'open_weights = {str(m["open_weights"]).lower()}')
    lines.append("")
    lines.append("[cost]")
    lines.append(f'input = {m["cost"]["input"]}')
    lines.append(f'output = {m["cost"]["output"]}')
    lines.append("")
    lines.append("[limit]")
    lines.append(f'context = {fmt_num(m["limit"]["context"])}')
    lines.append(f'output = {fmt_num(m["limit"]["output"])}')
    lines.append("")
    lines.append("[modalities]")
    inp = ", ".join(f'"{x}"' for x in m["modalities"]["input"])
    out = ", ".join(f'"{x}"' for x in m["modalities"]["output"])
    lines.append(f"input = [{inp}]")
    lines.append(f"output = [{out}]")
    return "\n".join(lines) + "\n"


def detect_changes(existing: dict | None, merged: dict) -> list[tuple[str, str, str]]:
    if not existing:
        return []
    changes = []

    def check(field, old, new):
        if str(old) != str(new):
            changes.append((field, str(old), str(new)))

    for cap in ("reasoning", "tool_call", "structured_output", "attachment"):
        check(cap, existing.get(cap), merged[cap])
    check("cost.input",    existing.get("cost", {}).get("input"),    merged["cost"]["input"])
    check("cost.output",   existing.get("cost", {}).get("output"),   merged["cost"]["output"])
    check("limit.context", existing.get("limit", {}).get("context"), merged["limit"]["context"])
    check("limit.output",  existing.get("limit", {}).get("output"),  merged["limit"]["output"])
    return changes


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()
    dry, verbose = args.dry_run, args.verbose
    prefix = "[DRY RUN] " if dry else ""

    print(f"{prefix}Fetching OVHcloud catalog...")
    catalog_list = fetch_json(CATALOG_LIST)
    llm_models = [m for m in catalog_list if m.get("category") in INCLUDE_CATEGORIES]
    print(f"  {len(llm_models)} LLM/Code/Visual/Guard models found\n")

    print(f"{prefix}Fetching /v1/models for max_completion_tokens...")
    try:
        v1_data = fetch_json(OVHCLOUD_API)
        v1_index = build_v1_index(v1_data.get("data", []))
        print(f"  {len(v1_index)} entries indexed\n")
    except Exception as e:
        print(f"  Warning: failed ({e}), output limits will fallback to context\n")
        v1_index = {}

    existing_files = {f.name for f in MODELS_DIR.glob("*.toml")} if MODELS_DIR.exists() else set()
    print(f"Found {len(existing_files)} existing TOML files\n")

    api_filenames = set()
    created = updated = unchanged = 0

    for entry in llm_models:
        catalog_id = entry["id"]

        # Fetch detail
        detail = fetch_json(CATALOG_DETAIL.format(id=catalog_id))
        aliases = detail.get("aliases", [catalog_id])

        filename   = alias_to_filename(aliases)
        filepath   = MODELS_DIR / filename
        api_filenames.add(filename)

        caps   = caps_from_catalog(detail)
        cost   = price_from_catalog(detail)
        limits = find_limits(catalog_id, aliases, v1_index) or {"context": 0, "output": 0}

        if verbose:
            print(f"  {catalog_id}")
            print(f"    reasoning={caps['reasoning']} tool_call={caps['tool_call']} "
                  f"attachment={caps['attachment']} structured_output={caps['structured_output']}")
            print(f"    cost={cost['input']}/{cost['output']} "
                  f"context={limits['context']} output={limits['output']}")

        existing = load_existing(filepath)
        merged   = build_model(detail, caps, cost, limits, existing)
        content  = to_toml(merged)

        if existing is None:
            created += 1
            if dry:
                print(f"[DRY RUN] Would create: {filename}")
            else:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
                print(f"Created: {filename}")
        else:
            changes = detect_changes(existing, merged)
            if changes:
                updated += 1
                if dry:
                    print(f"[DRY RUN] Would update: {filename}")
                else:
                    filepath.write_text(content)
                    print(f"Updated: {filename}")
                for field, old, new in changes:
                    print(f"  {field}: {old} → {new}")
                if not verbose:
                    print()
            else:
                unchanged += 1

    orphaned = existing_files - api_filenames
    for f in sorted(orphaned):
        print(f"Warning: orphaned (rename or delete): {f}")

    print()
    print(f"{prefix}Summary: {created} created, {updated} updated, "
          f"{unchanged} unchanged, {len(orphaned)} orphaned")


if __name__ == "__main__":
    main()
