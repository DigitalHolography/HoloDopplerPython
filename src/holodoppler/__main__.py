from __future__ import annotations

import sys

from holodoppler.cli import main as cli_main
from holodoppler.ui import UI


def main() -> int:
    if len(sys.argv) == 1:
        UI().mainloop()
        return 0

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())