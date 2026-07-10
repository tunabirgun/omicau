"""Environment-aware install hints for optional extras.

On a pipx-managed install (and on PEP 668 "externally managed" system Pythons such
as Debian / Ubuntu 24.04+), `pip install "omicau[extra]"` fails, so the missing-
dependency messages point at the command that works for the CURRENT install:
`pipx inject` when omicau lives in a pipx venv, otherwise the plain extra.
"""

from __future__ import annotations

import sys

# dep lists per extra, so `pipx inject omicau <packages>` reproduces the extra.
_EXTRA_PACKAGES = {
    "ui": "fastapi uvicorn python-multipart",
    "data": "requests google-cloud-storage",
    "llm": "anthropic openai",
    "yaml": "pyyaml",
    "cptac": "cptac",
}


def _in_pipx() -> bool:
    """True when omicau is running from a pipx-managed virtual environment."""
    prefix = sys.prefix.replace("\\", "/").lower()
    return "/pipx/venvs/" in prefix


def extra_hint(extra: str, packages: str | None = None) -> str:
    """Return the command to add the optional ``extra`` to this omicau install."""
    pkgs = packages or _EXTRA_PACKAGES.get(extra, extra)
    if _in_pipx():
        return f"pipx inject omicau {pkgs}"
    return f'pip install "omicau[{extra}]"'
