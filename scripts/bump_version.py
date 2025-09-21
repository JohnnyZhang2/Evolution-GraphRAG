#!/usr/bin/env python3
"""Version bump utility for Evolution RAG.

Features:
- Reads current api_version from graphrag/config/settings.py (regex parse).
- Supports bump types: major / minor / patch or explicit --set X.Y.Z.
- Updates settings.py in place.
- Updates CHANGELOG.md: moves Unreleased section content into new version block with date.
- Leaves an empty Unreleased template if missing.
- Adds compare link placeholders (can be replaced by real repo URL later).
- Writes a summary of changes to stdout.

Usage:
  python scripts/bump_version.py --type minor
  python scripts/bump_version.py --set 1.2.0
  python scripts/bump_version.py --dry-run --type patch

Exit codes:
 0 success
 1 generic failure
"""
from __future__ import annotations
import re
import sys
import argparse
from datetime import date
from pathlib import Path

SETTINGS_PATH = Path('graphrag/config/settings.py')
CHANGELOG_PATH = Path('CHANGELOG.md')

VERSION_REGEX = re.compile(r'api_version:\s*str\s*=\s*Field\(\"([0-9]+\.[0-9]+\.[0-9]+)\"')

UNRELEASED_TEMPLATE = """## [Unreleased]\n\n### Added\n\n- (placeholder)\n\n### Changed\n\n- (placeholder)\n\n### Deprecated\n\n- (placeholder)\n\n### Removed\n\n- (placeholder)\n\n### Fixed\n\n- (placeholder)\n\n### Security\n\n- (placeholder)\n\n"""

COMPARE_LINK_FOOTER = """[Unreleased]: https://example.com/compare/{new_version}...HEAD\n[{new_version}]: https://example.com/releases/{new_version}\n"""

def parse_args():
    p = argparse.ArgumentParser(description='Bump api_version and update CHANGELOG.')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--type', choices=['major','minor','patch'], help='Bump type')
    group.add_argument('--set', dest='set_version', help='Explicit version X.Y.Z')
    p.add_argument('--dry-run', action='store_true', help='Show changes without writing files')
    return p.parse_args()


def read_current_version() -> str:
    text = SETTINGS_PATH.read_text(encoding='utf-8')
    m = VERSION_REGEX.search(text)
    if not m:
        raise RuntimeError('api_version not found in settings.py')
    return m.group(1)


def compute_new_version(current: str, bump_type: str | None, explicit: str | None) -> str:
    if explicit:
        if not re.match(r'^\d+\.\d+\.\d+$', explicit):
            raise ValueError('Explicit version must be X.Y.Z')
        return explicit
    major, minor, patch = map(int, current.split('.'))
    if bump_type == 'major':
        return f"{major+1}.0.0"
    if bump_type == 'minor':
        return f"{major}.{minor+1}.0"
    if bump_type == 'patch':
        return f"{major}.{minor}.{patch+1}"
    raise ValueError('Invalid bump type')


def update_settings_file(old: str, new: str, dry: bool):
    text = SETTINGS_PATH.read_text(encoding='utf-8')
    new_text = VERSION_REGEX.sub(lambda m: m.group(0).replace(old, new), text, count=1)
    if text == new_text:
        raise RuntimeError('Failed to replace version in settings.py')
    if not dry:
        SETTINGS_PATH.write_text(new_text, encoding='utf-8')


def extract_unreleased(changelog: str) -> tuple[str, str]:
    # Capture Unreleased block until next '## [' or end
    pattern = re.compile(r'## \[Unreleased\](.*?)(?:\n## \[|\Z)', re.DOTALL)
    m = pattern.search(changelog)
    if not m:
        return '', changelog
    block = m.group(1).rstrip('\n')
    # Remove the block including header
    start, end = m.span()
    remaining = changelog[:start] + changelog[end-4:]  # keep the starting '## [' of next section if exists
    return block.strip('\n'), remaining


def insert_new_version(changelog_remaining: str, new_version: str, unreleased_content: str) -> str:
    today = date.today().isoformat()
    if unreleased_content.strip():
        body = unreleased_content
    else:
        body = '### Added\n\n- No changes recorded.\n'
    # Ensure we don't duplicate existing Unreleased template
    if '## [Unreleased]' not in changelog_remaining:
        changelog_remaining = changelog_remaining.strip() + '\n\n' + UNRELEASED_TEMPLATE
    new_block = f"## [{new_version}] - {today}\n\n{body}\n\n"
    # Insert after Unreleased section (which we will have just recreated at top after header lines)
    parts = changelog_remaining.split('## [Unreleased]')
    header = parts[0].rstrip() + '\n\n'
    rest = '## [Unreleased]'.join(parts[1:])
    return header + '## [Unreleased]\n\n' + rest.lstrip() + '\n' + new_block


def update_compare_links(changelog: str, new_version: str) -> str:
    # Remove old Unreleased link lines if exist and append new
    lines = [l for l in changelog.rstrip().splitlines() if not re.match(r'^\[Unreleased\]:', l)]
    # Remove any line that matches [X.Y.Z]: placeholder with same new_version to avoid duplicates
    lines = [l for l in lines if not re.match(fr'^\[{re.escape(new_version)}\]:', l)]
    lines.append(COMPARE_LINK_FOOTER.format(new_version=new_version).rstrip())
    return '\n'.join(lines) + '\n'


def update_changelog(new_version: str, dry: bool):
    if not CHANGELOG_PATH.exists():
        raise RuntimeError('CHANGELOG.md not found')
    original = CHANGELOG_PATH.read_text(encoding='utf-8')
    unreleased, remaining = extract_unreleased(original)
    updated = insert_new_version(remaining, new_version, unreleased)
    updated = update_compare_links(updated, new_version)
    if not dry:
        CHANGELOG_PATH.write_text(updated, encoding='utf-8')
    return unreleased, updated


def main():
    args = parse_args()
    try:
        current = read_current_version()
        new_version = compute_new_version(current, args.type, args.set_version)
        unreleased_content, updated_changelog = update_changelog(new_version, args.dry_run)
        update_settings_file(current, new_version, args.dry_run)
        print(f"Current version: {current}")
        print(f"New version: {new_version}{' (dry-run)' if args.dry_run else ''}")
        if unreleased_content.strip():
            print('\nUnreleased content moved:')
            print(unreleased_content[:500] + ('...' if len(unreleased_content) > 500 else ''))
        else:
            print('\nNo unreleased content found (template may have been empty).')
        if args.dry_run:
            print('\n--- DRY RUN CHANGELOG PREVIEW (first 300 chars) ---')
            print(updated_changelog[:300])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
