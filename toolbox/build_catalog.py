#!/usr/bin/env python3
"""Build and validate the machine-readable experience catalog.

Usage:
    python3 toolbox/build_catalog.py           # rewrite catalog.json
    python3 toolbox/build_catalog.py --check   # fail if catalog.json is stale/invalid

The script intentionally uses only the Python standard library so that it works
in a bare checkout and in CI without installing anything.

Frontmatter is a deliberately small YAML subset:

    ---
    name: kebab-case-slug
    title: 中文标题
    description: 一句话说明什么时候读这条经验。
    category: compile
    tags: [triton, ptxas]
    symptoms:
      - 现象一
      - 现象二
    stacks: [triton]
    updated: 2026-11-02
    ---

Scalars may be quoted with double quotes when they contain a colon. Block
scalars (``|``/``>``) and nested mappings are not supported on purpose: the
catalog is data, not prose storage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIENCES_DIR = ROOT / "experiences"
CATALOG_PATH = ROOT / "catalog.json"
INDEX_PATH = EXPERIENCES_DIR / "README.md"

CATEGORY_ORDER = [
    "methodology",
    "compile",
    "crash",
    "correctness",
    "performance",
    "engineering",
]

REQUIRED_FIELDS = [
    "name",
    "title",
    "description",
    "category",
    "tags",
    "symptoms",
    "stacks",
    "updated",
]

LIST_FIELDS = {"tags", "symptoms", "stacks"}
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CatalogError(RuntimeError):
    """Raised when an entry cannot be parsed or does not meet the schema."""


def _parse_scalar(raw: str):
    value = raw.strip()
    if not value:
        return ""
    if value[0] in "\"'":
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid input
            raise CatalogError(f"cannot parse quoted scalar {value!r}: {exc}") from exc
    return value


def _split_inline_list(body: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in body:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == ",":
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    if quote:
        raise CatalogError(f"unterminated quote in inline list: {body!r}")
    items.append("".join(current))
    return [_parse_scalar(item) for item in items if item.strip()]


def parse_frontmatter(text: str, source: Path) -> dict:
    if not text.startswith("---\n"):
        raise CatalogError(f"{source}: file must start with '---' frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise CatalogError(f"{source}: frontmatter is not closed with '---'")

    data: dict = {}
    active_key: str | None = None
    for raw_line in text[4:end].splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") or line.startswith("- "):
            if active_key is None:
                raise CatalogError(f"{source}: list item without a key: {line!r}")
            if active_key not in data or not isinstance(data[active_key], list):
                data[active_key] = []
            data[active_key].append(_parse_scalar(line.split("- ", 1)[1]))
            continue
        key, sep, rest = line.partition(":")
        if not sep:
            raise CatalogError(f"{source}: expected 'key: value', got {line!r}")
        key = key.strip()
        rest = rest.strip()
        active_key = key
        if rest == "":
            data[key] = []
        elif rest.startswith("[") and rest.endswith("]"):
            data[key] = _split_inline_list(rest[1:-1])
        else:
            data[key] = _parse_scalar(rest)
    return data


def load_entries() -> list[dict]:
    entries: list[dict] = []
    seen_names: set[str] = set()
    if not EXPERIENCES_DIR.is_dir():
        raise CatalogError(f"{EXPERIENCES_DIR} does not exist")

    for path in sorted(EXPERIENCES_DIR.rglob("*.md")):
        if path == INDEX_PATH or path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        meta = parse_frontmatter(text, path)
        relative = path.relative_to(ROOT)

        for field in REQUIRED_FIELDS:
            if field not in meta:
                raise CatalogError(f"{relative}: missing required field '{field}'")
            if meta[field] in ("", [], None):
                raise CatalogError(f"{relative}: field '{field}' must not be empty")

        name = str(meta["name"])
        if not SLUG_RE.match(name):
            raise CatalogError(f"{relative}: name {name!r} is not kebab-case")
        if name != path.stem:
            raise CatalogError(
                f"{relative}: name {name!r} must match the file name {path.stem!r}"
            )
        if name in seen_names:
            raise CatalogError(f"{relative}: duplicate name {name!r}")
        seen_names.add(name)

        category = str(meta["category"])
        if category not in CATEGORY_ORDER:
            raise CatalogError(
                f"{relative}: category {category!r} is not one of {CATEGORY_ORDER}"
            )

        for field in LIST_FIELDS:
            value = meta[field]
            if not isinstance(value, list):
                raise CatalogError(f"{relative}: field '{field}' must be a list")
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise CatalogError(f"{relative}: field '{field}' has an empty item")

        updated = str(meta["updated"])
        if not DATE_RE.match(updated):
            raise CatalogError(f"{relative}: updated must be YYYY-MM-DD")

        description = str(meta["description"])
        if not 30 <= len(description) <= 500:
            raise CatalogError(
                f"{relative}: description must be 30-500 characters "
                f"(currently {len(description)})"
            )

        entries.append(
            {
                "name": name,
                "title": str(meta["title"]),
                "description": description,
                "category": category,
                "tags": [str(item) for item in meta["tags"]],
                "symptoms": [str(item) for item in meta["symptoms"]],
                "stacks": [str(item) for item in meta["stacks"]],
                "updated": updated,
                "path": relative.as_posix(),
            }
        )

    entries.sort(key=lambda entry: (CATEGORY_ORDER.index(entry["category"]), entry["name"]))
    return entries


def build_catalog(entries: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "count": len(entries),
        "categories": CATEGORY_ORDER,
        "entries": entries,
    }


def check_index(entries: list[dict]) -> list[str]:
    problems: list[str] = []
    if not INDEX_PATH.is_file():
        return [f"{INDEX_PATH.relative_to(ROOT)} is missing"]
    index_text = INDEX_PATH.read_text(encoding="utf-8")
    for entry in entries:
        if entry["path"] not in index_text and Path(entry["path"]).name not in index_text:
            problems.append(
                f"{entry['path']} is not linked from {INDEX_PATH.relative_to(ROOT)}"
            )
    return problems


def render(catalog: dict) -> str:
    return json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate entries and fail if catalog.json or the index is stale",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_entries()
        problems = check_index(entries)
        catalog = build_catalog(entries)
        rendered = render(catalog)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    if args.check:
        if not CATALOG_PATH.is_file():
            print("error: catalog.json is missing; run the script without --check", file=sys.stderr)
            return 1
        current = CATALOG_PATH.read_text(encoding="utf-8")
        if current != rendered:
            print("error: catalog.json is stale; run the script without --check", file=sys.stderr)
            return 1
        print(f"ok: {len(entries)} entries, catalog and index are up to date")
        return 0

    CATALOG_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {CATALOG_PATH.relative_to(ROOT)} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
