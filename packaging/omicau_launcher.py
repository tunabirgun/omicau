"""Frozen-app entry point.

Double-clicking the packaged app (no CLI args) launches the local web UI and
opens a browser; passing CLI args behaves exactly like the ``omicau`` command
(run / bootstrap / verify / check-env / ui). ``multiprocessing.freeze_support``
is required so any worker processes behave under PyInstaller.
"""

import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()
    if len(sys.argv) == 1:                 # double-click -> open the UI
        sys.argv.append("ui")
    from omicau.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
