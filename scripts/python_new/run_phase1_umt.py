#!/usr/bin/env python3
"""Compatibility wrapper for phase-1 UMT training."""

from scripts.python_new.run_single import main


if __name__ == "__main__":
    main("phase1", umt=True)
