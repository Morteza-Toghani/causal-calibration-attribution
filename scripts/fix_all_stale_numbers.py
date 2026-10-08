from pathlib import Path

base = Path("docs/paper1_draft")
FILES = [
    "01_abstract.md",
    "02_introduction.md",
    "06_results.md",
    "07_discussion.md",
]

# Ordered replacements: longer/specific first, then general
REPL = [
    # Numerical substitutions (stale → current)
    ("0.07955", "0.07749"),
    ("0.15449", "0.15268"),
    ("0.01111", "0.01194"),
    ("0.3725",  "0.3743"),
    ("\u22120.02407", "\u22120.02418"),   # minus sign + number
    ("\u22120.03256", "\u22120.03307"),
]

report = []
for name in FILES:
    p = base / name
    s = p.read_text(encoding="utf-8")
    orig = s
    for old, new in REPL:
        s = s.replace(old, new)
    # Fix the "14 and 234 times" claim in 06
    s = s.replace(
        "The observation effect was between about 14 and 234 times "
        "larger than the dynamics or policy effect.",
        "The observation effect was between about 15 and 155 times "
        "larger than the dynamics or policy effect.",
    )
    if s != orig:
        p.write_text(s, encoding="utf-8", newline="\n")
        report.append(f"{name}: updated")
    else:
        report.append(f"{name}: NO CHANGE")

print("\n".join(report))
