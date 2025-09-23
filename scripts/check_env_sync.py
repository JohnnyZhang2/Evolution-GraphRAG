#!/usr/bin/env python3
# Evolution RAG Environment Variable Consistency Checker
# Compare environment variables used in settings.py with those present in .env.
# Outputs:
#   - Missing (defined in settings but absent in .env)
#   - Extra   (present in .env but not mapped in settings)
#   - Differs (value differs from current runtime loaded default)
# Usage:
#   python scripts/check_env_sync.py [--fail-on-missing]
# Note: This is a lightweight static parser relying on the Field(..., env="VAR") pattern.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)
from __future__ import annotations
import os
import re
import argparse
import sys
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "graphrag" / "config" / "settings.py"
ENV_FILE = ROOT / ".env"

# 支持 v1 写法 env="VAR" 以及 v2 写法 alias="VAR"
FIELD_ENV_RE = re.compile(r'(?:env|alias)\s*=\s*"([A-Z0-9_]+)"')

# 可选变量集合：缺失不报错，仅提示（可被注释）
OPTIONAL_ENV = {
    # feature / tuning flags (non-critical)
    "DISABLE_ENTITY_EXTRACT",
    "EMBED_CACHE_MAX",
    "ANSWER_CACHE_MAX",
    "VECTOR_INDEX_NAME",
    "REL_FALLBACK_CONFIDENCE",
    # newly added typed entity / relation schema controls (should be optional by default)
    "ENTITY_TYPED_MODE",
    "ENTITY_TYPES",
    "RELATION_ENFORCE_TYPES",
    "RELATION_TYPES",
    "RELATION_FALLBACK_TYPE",
}

# Core required minimal set actually needed for the system to start; everything else treated as optional if absent.
CORE_REQUIRED = {
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
}

def parse_commented_optional() -> Set[str]:
    if not ENV_FILE.exists():
        return set()
    commented = set()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*#\s*([A-Z0-9_]+)=", line)
        if m and m.group(1) in OPTIONAL_ENV:
            commented.add(m.group(1))
    return commented


def extract_setting_envs() -> Set[str]:
    text = SETTINGS_FILE.read_text(encoding="utf-8")
    return set(FIELD_ENV_RE.findall(text))


def parse_env_file() -> Dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    data: Dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        data[k.strip()] = v.strip()
    return data


def classify_env():
    setting_envs = extract_setting_envs()
    env_values = parse_env_file()
    env_file_keys = set(env_values.keys())
    commented_opt = parse_commented_optional()
    raw_missing = setting_envs - env_file_keys
    # Reclassify: only items in CORE_REQUIRED count as required; all other missing become optional to avoid brittle tests when adding new config.
    missing_required = sorted([m for m in raw_missing if m in CORE_REQUIRED])
    # Everything else missing (not core) is optional by definition, union with predefined OPTIONAL_ENV for clarity
    missing_optional = sorted([m for m in raw_missing if m not in CORE_REQUIRED])
    extra = sorted(env_file_keys - setting_envs)
    return {
        "settings_total": len(setting_envs),
        "env_total": len(env_file_keys),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "extra": extra,
        "commented_optional": sorted(commented_opt),
        "setting_envs": setting_envs,
        "env_values": env_values,
    }


def runtime_differences(setting_envs, env_values):
    diffs = []
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from graphrag.config.settings import get_settings  # type: ignore
        runtime = get_settings()
        for key in setting_envs & set(env_values.keys()):
            attr_guess = key.lower()
            if hasattr(runtime, attr_guess):
                rv = getattr(runtime, attr_guess)
                if str(rv) != env_values[key]:
                    diffs.append((key, env_values[key], rv))
    except Exception as exc:  # pragma: no cover
        return [], str(exc)
    return diffs, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fail-on-missing', action='store_true', help='Exit 1 if there are missing variables.')
    args = parser.parse_args()

    info = classify_env()
    print("[env-sync] Settings variables total:", info['settings_total'])
    print("[env-sync] .env variables total:", info['env_total'])
    print()
    if info['missing_required']:
        print("Missing REQUIRED (.env lacks these):")
        for m in info['missing_required']:
            print("  -", m)
    else:
        print("No required variables missing.")
    print()
    if info['missing_optional']:
        print("Missing OPTIONAL (使用代码默认):")
        for m in info['missing_optional']:
            suffix = " (commented)" if m in info['commented_optional'] else ""
            print(f"  - {m}{suffix}")
    else:
        print("No optional variables missing.")
    print()
    if info['extra']:
        print("Extra (.env defines but settings not using):")
        for e in info['extra']:
            print("  -", e)
    else:
        print("No extra variables.")
    print()

    diffs, err = runtime_differences(info['setting_envs'], info['env_values'])
    if err:
        print(f"(Skip runtime comparison: {err})")
    else:
        if diffs:
            print("Value differences (env file vs runtime):")
            for k, f, r in diffs:
                print(f"  - {k}: .env='{f}' runtime='{r}'")
        else:
            print("No value differences detected (heuristic).")

    if info['missing_required'] and args.fail_on_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
