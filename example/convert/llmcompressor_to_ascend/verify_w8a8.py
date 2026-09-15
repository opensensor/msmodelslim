"""Verify an Ascend W8A8_DYNAMIC export against its compressed-tensors source."""

import argparse
import json
from pathlib import Path

from msmodelslim.core.quant_service.modelslim_convert.impl.int8_verify import verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--export", dest="destination", required=True, type=Path)
    parser.add_argument(
        "--check-values", action="store_true", help="Read every tensor in CPU chunks and compare values"
    )
    parser.add_argument("--chunk-rows", type=int, default=1024)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.source, args.destination, check_values=args.check_values, chunk_rows=args.chunk_rows), indent=2
        )
    )
