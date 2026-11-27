"""Check unrounded statement and branch coverage separately."""
import json
from pathlib import Path

totals = json.loads(Path("coverage.json").read_text())["totals"]
for name, hit, total, minimum in (
    ("Statements", "covered_lines", "num_statements", 95),
    ("Branches", "covered_branches", "num_branches", 90),
):
    covered, count = totals[hit], totals[total]
    print(f"{name}: {covered}/{count} ({100*covered/count:.2f}%)")
    if 100*covered < minimum*count:
        raise SystemExit(f"{name} below {minimum}%")
