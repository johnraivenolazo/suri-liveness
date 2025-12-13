"""Training script wrapper for backward compatibility."""

from src.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
