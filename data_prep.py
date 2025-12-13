"""Data preparation script wrapper for backward compatibility."""

from src.infra.data_prep import main

if __name__ == "__main__":
    raise SystemExit(main())
