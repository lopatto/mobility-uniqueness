#!/usr/bin/env python3
"""Run all certified numerical checks and record a complete verification log."""

from __future__ import annotations

import contextlib
import io
import platform
import sys
import traceback
from pathlib import Path

OUTPUT_FILE = Path(__file__).with_name("verification_output.txt")


def _python_flint_version() -> str:
    try:
        import flint
    except Exception as exc:  # reported again by the actual verifier import below
        return f"unavailable ({type(exc).__name__}: {exc})"
    return str(getattr(flint, "__version__", "unknown"))


def _underlying_flint_version() -> str:
    """Report the version of the FLINT library used by python-flint.

    Arb is part of FLINT in the versions supported by python-flint 0.9.0, so
    the FLINT runtime version identifies the Arb implementation used here.
    """
    try:
        import flint
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"

    value = getattr(flint, "__FLINT_VERSION__", None)
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _metadata(exit_status: int) -> str:
    return "\n".join(
        [
            "Certified numerical verification",
            "================================",
            f"Python version: {sys.version.replace(chr(10), ' ')}",
            f"Python implementation: {platform.python_implementation()}",
            f"python-flint version: {_python_flint_version()}",
            f"FLINT/Arb backend version: {_underlying_flint_version()}",
            f"Exit status: {exit_status}",
            "",
        ]
    )


def _run_verifiers() -> None:
    import verify_certificate

    verify_certificate.main()


def main() -> int:
    body = io.StringIO()
    exit_status = 0

    with contextlib.redirect_stdout(body), contextlib.redirect_stderr(body):
        try:
            _run_verifiers()
        except BaseException:  # preserve a complete traceback in the verification log
            exit_status = 1
            print("\nVERIFICATION FAILED\n")
            traceback.print_exc()

    text = _metadata(exit_status) + body.getvalue()
    OUTPUT_FILE.write_text(text, encoding="utf-8")

    # Mirror the complete file to the terminal so interactive use is unchanged.
    print(text, end="")
    print(f"\nComplete verification log written to: {OUTPUT_FILE}")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
