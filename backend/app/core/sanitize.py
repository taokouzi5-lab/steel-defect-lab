"""Control-character scrubbing for values echoed into errors and logs (L1).

Client-controlled identifiers (sample ids, model ids) may contain newlines
or terminal escape sequences; echoing them verbatim into error messages and
log lines enables log injection. Strip C0 control characters (and DEL)
before interpolation.
"""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_text(value: object) -> str:
    """Return ``str(value)`` with C0 control characters and DEL removed."""
    return _CONTROL_CHARS.sub("", str(value))
