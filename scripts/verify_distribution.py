"""Compare both local distribution digests with the public PyPI release."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

version = Path("VERSION").read_text().strip()
with urlopen(f"https://pypi.org/pypi/shotweave/{version}/json", timeout=30) as response:
    published = {item["filename"]: item["digests"]["sha256"] for item in json.load(response)["urls"]}
local = {p.name: sha256(p.read_bytes()).hexdigest() for p in Path("dist").iterdir()}
if len(local) != 2 or published != local:
    raise SystemExit("Distribution digest mismatch")
for name, digest in sorted(local.items()):
    print(name, digest)
