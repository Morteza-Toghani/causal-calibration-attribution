"""Extract [REF: ...], [FILL], [VERIFY], [TODO] placeholders from drafts.

Outputs:
  docs/paper1_draft/_PLACEHOLDERS.md   (human-readable checklist)
  docs/paper1_draft/_PLACEHOLDERS.json (machine-readable)
"""
import re
import json
from pathlib import Path

DRAFT_DIR = Path("docs/paper1_draft")
PATTERNS = {
    "REF":    re.compile(r"\[REF:\s*([^\]]+)\]"),
    "FILL":   re.compile(r"\[FILL(?::\s*([^\]]+))?\]"),
    "VERIFY": re.compile(r"\[VERIFY(?::\s*([^\]]+))?\]"),
    "TODO":   re.compile(r"\[TODO(?::\s*([^\]]+))?\]"),
    "CITE":   re.compile(r"\[CITE:\s*([^\]]+)\]"),
    "FIGURE": re.compile(r"\[FIG(?:URE)?:\s*([^\]]+)\]"),
    "TABLE":  re.compile(r"\[TABLE:\s*([^\]]+)\]"),
}

records = []
for md in sorted(DRAFT_DIR.glob("*.md")):
    if md.name.startswith("_") or md.name.startswith("VERIFY_"):
        continue
    text = md.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, pat in PATTERNS.items():
            for m in pat.finditer(line):
                records.append({
                    "file": md.name,
                    "line": line_no,
                    "kind": kind,
                    "content": (m.group(1) or "").strip(),
                    "raw": m.group(0),
                    "context": line.strip()[:160],
                })

# group by kind
by_kind = {}
for r in records:
    by_kind.setdefault(r["kind"], []).append(r)

# write markdown checklist
out_md = ["# Placeholder checklist — Paper 1 drafts\n",
          f"Total placeholders: **{len(records)}**\n"]
for kind in ["REF", "CITE", "FILL", "VERIFY", "TODO", "FIGURE", "TABLE"]:
    items = by_kind.get(kind, [])
    if not items:
        continue
    out_md.append(f"\n## {kind} ({len(items)})\n")
    for r in items:
        out_md.append(f"- [ ] **{r['file']}:{r['line']}** — `{r['raw']}`")
        out_md.append(f"      context: {r['context']}")
out_md.append("")

Path(DRAFT_DIR / "_PLACEHOLDERS.md").write_text(
    "\n".join(out_md), encoding="utf-8", newline="\n"
)
Path(DRAFT_DIR / "_PLACEHOLDERS.json").write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8", newline="\n"
)

# console summary
print(f"Scanned {len(list(DRAFT_DIR.glob('*.md')))} files in {DRAFT_DIR}")
print(f"Found {len(records)} placeholders total")
for kind in ["REF", "CITE", "FILL", "VERIFY", "TODO", "FIGURE", "TABLE"]:
    n = len(by_kind.get(kind, []))
    if n:
        print(f"  {kind:8s}: {n}")
print("\nWrote:")
print(f"  {DRAFT_DIR / '_PLACEHOLDERS.md'}")
print(f"  {DRAFT_DIR / '_PLACEHOLDERS.json'}")
