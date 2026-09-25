#!/usr/bin/env python3
"""Run all certified numerical checks and record a complete verification log."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import io
import platform
import subprocess
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
    """Best-effort report of the loaded FLINT/Arb backend version.

    Arb is part of FLINT in the versions supported by python-flint 0.9.0, so
    the FLINT runtime version identifies the Arb implementation used here.
    """
    try:
        import flint
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"

    # Use a public Python-level version attribute if python-flint exposes one.
    for name in ("__flint_version__", "flint_version", "FLINT_VERSION"):
        value = getattr(flint, name, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if isinstance(value, bytes):
            value = value.decode("ascii", "replace")
        if isinstance(value, str) and value:
            return value

    # FLINT 3 exports the runtime string `flint_version`.  Try the already
    # loaded process first, and then the system-resolved libflint.
    candidates: list[str | None] = [None]
    libname = ctypes.util.find_library("flint")
    if libname:
        candidates.append(libname)
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate) if candidate is not None else ctypes.CDLL(None)
            value = ctypes.c_char_p.in_dll(lib, "flint_version").value
            if value:
                return value.decode("ascii", "replace")
        except Exception:
            pass

    # Last-resort diagnostic.  This does not affect the verification itself.
    try:
        proc = subprocess.run(
            ["pkg-config", "--modversion", "flint"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = proc.stdout.strip()
        if value:
            return value + " (reported by pkg-config)"
    except Exception:
        pass

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
    import verify_scalar_certificates
    import verify_modulus_certificates

    verify_scalar_certificates.main()
    print()
    verify_modulus_certificates.main()


def main() -> int:
    body = io.StringIO()
    exit_status = 0

    with contextlib.redirect_stdout(body), contextlib.redirect_stderr(body):
        try:
            _run_verifiers()
        except BaseException:  # preserve a complete traceback in the audit log
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
