#!/usr/bin/env python
"""Check that required packages in requirements.txt are actually importable.

Features:
- Parse requirements.txt lines (ignores version specifiers, extras and comments)
- Categorize packages into required vs optional (simple heuristic list you can adjust)
- Attempt to import top-level modules (mapping some known package->module name differences)
- Report missing imports and optionally output JSON (--json)
- Expose a check_imports() function for pytest

Exit codes:
0 = all required importable
1 = missing required packages
2 = requirements.txt not found / other error

Usage:
python scripts/check_requirements_imports.py
python scripts/check_requirements_imports.py --json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REQ_FILE = os.path.join(REPO_ROOT, 'requirements.txt')

# Heuristic classification: adjust as project evolves
OPTIONAL_PACKAGES = {
    'pytest',  # test only
    'python-dotenv',  # if later added
    'scikit-learn',  # treat as optional (only needed if advanced rerank/ML features enabled)
}

# Mapping requirement name -> import module names to try
IMPORT_NAME_OVERRIDES: Dict[str, List[str]] = {
    'pypdf': ['pypdf', 'PyPDF2'],  # allow fallback if old package still installed
    'python-docx': ['docx'],
    'openpyxl': ['openpyxl'],
    'neo4j': ['neo4j'],
    'pydantic-settings': ['pydantic_settings'],
}

PKG_NAME_REGEX = re.compile(r'^[A-Za-z0-9_.-]+')

@dataclass
class ImportResult:
    name: str
    category: str  # required | optional
    importable: bool
    tried: List[str]
    error: str | None = None

@dataclass
class CheckReport:
    required_total: int
    required_missing: int
    optional_total: int
    optional_missing: int
    results: List[ImportResult]

    def summary(self) -> str:
        return (f"Required missing: {self.required_missing}/{self.required_total}; "
                f"Optional missing: {self.optional_missing}/{self.optional_total}")

def parse_requirements(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pkgs: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = PKG_NAME_REGEX.match(line)
            if not m:
                continue
            name = m.group(0).lower()
            pkgs.append(name)
    return sorted(set(pkgs))

def classify(name: str) -> str:
    return 'optional' if name in OPTIONAL_PACKAGES else 'required'

def import_attempts(name: str) -> Tuple[List[str], ImportResult]:
    module_names = IMPORT_NAME_OVERRIDES.get(name, [name.replace('-', '_')])
    errors = []
    for mod in module_names:
        try:
            __import__(mod)
            return module_names, ImportResult(name=name, category=classify(name), importable=True, tried=module_names)
        except Exception as e:  # noqa: BLE001
            errors.append((mod, e))
    # If none succeeded
    err_msg = '; '.join(f"{mod}: {type(e).__name__}: {e}" for mod, e in errors)
    return module_names, ImportResult(name=name, category=classify(name), importable=False, tried=module_names, error=err_msg)

def check_imports(requirements_file: str = REQ_FILE) -> CheckReport:
    try:
        packages = parse_requirements(requirements_file)
    except FileNotFoundError:
        raise
    results: List[ImportResult] = []
    for pkg in packages:
        _, res = import_attempts(pkg)
        results.append(res)
    req_missing = sum(1 for r in results if r.category == 'required' and not r.importable)
    opt_missing = sum(1 for r in results if r.category == 'optional' and not r.importable)
    req_total = sum(1 for r in results if r.category == 'required')
    opt_total = sum(1 for r in results if r.category == 'optional')
    return CheckReport(
        required_total=req_total,
        required_missing=req_missing,
        optional_total=opt_total,
        optional_missing=opt_missing,
        results=results,
    )

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Verify importability of dependencies.')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args(argv)

    try:
        report = check_imports()
    except FileNotFoundError:
        print(f"requirements.txt not found at {REQ_FILE}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            'summary': report.summary(),
            'required_missing': report.required_missing,
            'optional_missing': report.optional_missing,
            'results': [asdict(r) for r in report.results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        for r in report.results:
            status = 'OK' if r.importable else 'MISSING'
            print(f"- {r.name:<25} {status} ({r.category}) tried={r.tried}" + (f" -> {r.error}" if r.error else ''))

    return 0 if report.required_missing == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
