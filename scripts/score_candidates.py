#!/usr/bin/env python3
"""Placeholder for shared post-hoc candidate scoring.

The first implementation should consume candidate JSONL from B0/B1/B2/B3 and
emit a common metric schema: pLDDT, iPAE, ipTM, pTM, cross-oracle agreement,
sequence sanity metrics, and resource accounting.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raise NotImplementedError(
        "Shared scoring is planned after the dry-run provenance schema is verified. "
        f"Received candidates={args.candidates} output={args.output}."
    )


if __name__ == "__main__":
    main()
