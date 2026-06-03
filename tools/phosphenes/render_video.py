from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.pipeline import main as _pipeline_main


def main() -> None:
    if not any(arg == "--media-type" or arg.startswith("--media-type=") for arg in sys.argv):
        sys.argv.extend(["--media-type", "video"])
    _pipeline_main()


if __name__ == "__main__":
    main()
