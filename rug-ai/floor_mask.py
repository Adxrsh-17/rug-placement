"""Legacy wrapper: use run_assignment.py instead of notebook scripts."""

from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from run_assignment import main as cli_main

    sys.argv = [
        "run_assignment.py",
        "--mode",
        "single",
        "--room",
        "room1.jpeg",
        "--rug",
        "rug1.jpg",
    ]
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
