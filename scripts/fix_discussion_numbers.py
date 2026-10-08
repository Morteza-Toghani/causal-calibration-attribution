from pathlib import Path

p = Path("docs/paper1_draft/07_discussion.md")
s = p.read_text(encoding="utf-8")
orig = s

# Fix the magnitudes in the pre-recalibration example
s = s.replace(
    "for example, 0.03256 vs 0.02407 for Walker2d at high severity",
    "for example, 0.03307 vs 0.02418 for Walker2d at high severity",
)

# Fix the ratio range claim
s = s.replace(
    "the observation effect exceeded the dynamics effect by factors of "
    "14 to 178 after recalibration",
    "the observation effect exceeded the dynamics effect by factors of "
    "15 to 155 after recalibration",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    print("07_discussion.md: updated")
else:
    print("07_discussion.md: NO CHANGE")
