"""Estimate a weighted sum from JSON measurement data."""
import argparse
import json
import sys
from dataclasses import asdict

from .experiment import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                        help="JSON file with coefficients, shots and successes (default: stdin)")
    args = parser.parse_args()
    try:
        data = json.load(args.file)
        result = summarize(data["coefficients"], data["shots"], data["successes"])
        print(json.dumps(asdict(result), allow_nan=False))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.exit(2, f"Invalid measurement data: {exc}\n")


if __name__ == "__main__":
    main()
