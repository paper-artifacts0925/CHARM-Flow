import argparse
import os
import pickle
import tempfile
from pathlib import Path
import sys

WORK_ROOT = Path(__file__).resolve().parents[2]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from src.models.hungarian_flow.context_artifact import (
    load_recursive_context_bank,
    sha256_file,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a legacy pickle runtime view from a safe context NPZ"
    )
    parser.add_argument("--context-npz", type=Path, required=True)
    parser.add_argument("--expected-context-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source = args.context_npz.expanduser().resolve()
    expected = str(args.expected_context_sha256).lower()
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(
            f"canonical context SHA256 mismatch: expected {expected}, got {actual}"
        )
    bank = load_recursive_context_bank(source)
    bank["metadata"] = dict(bank["metadata"])
    bank["metadata"].update(
        {
            "canonical_safe_npz": str(source),
            "canonical_safe_npz_sha256": actual,
            "runtime_compatibility_format": "trusted-local-pickle",
        }
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(bank, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(f"canonical_npz_sha256={actual}")
    print(f"legacy_pickle={output}")
    print(f"legacy_pickle_sha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
