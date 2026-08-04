#!/usr/bin/env python3
"""Compatibility wrapper for UMT training in the unified CLP runner."""

from scripts.python_new.run_single import main


if __name__ == "__main__":
    main("all_at_once", umt=True)
