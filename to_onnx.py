"""ONNX export script wrapper for backward compatibility."""

from src.infra.export_onnx import main

if __name__ == "__main__":
    raise SystemExit(main())
