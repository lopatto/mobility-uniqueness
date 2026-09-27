#!/usr/bin/env python3
"""Rigorous Arb verification of the numerical inequality in Lemma 5.2 of
P. Lopatto, "Uniqueness of the mobility edge for Lévy matrices".

Tested with python-flint 0.9.0.  All interval endpoints are exact rational
numbers.  The program evaluates only the explicit finite formula stated in
the manuscript and recursively bisects an interval whenever direct ball
evaluation does not yet prove strict positivity.

The certified inequality (the unnumbered display in the proof of Lemma 5.2
of the paper; by Lemma 4.5 it implies eq. (47)) is

    g_{1,11}(c_8) - 2 / (c_3 (q + s))^2 > 0,        1/6 <= alpha <= 1,

where s = sin(pi alpha/2), q = sqrt(1 + s^2),

    c_3     = pi Gamma(1 + alpha/2) / (Gamma(1 - alpha/2) Gamma((1 + alpha)/2)^2),
    c_6     = 2 + (5 s + s^2)/4,
    c_7     = Gamma(2 alpha) / (3 Gamma(alpha)),
    c_8     = sqrt((1 - 2/c_6) / (c_7 s^2)),
    g_{1,N}(u) = (alpha/2) u^2 sum_{k=0}^{N} (-u)^k Gamma((k+2) alpha/2) / k!.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Callable, Sequence

from flint import arb, ctx

ctx.prec = 256

Q = Fraction
ZERO = arb(0)
ONE = arb(1)
HALF = arb("1/2")


@dataclass
class Certificate:
    name: str
    boxes: int
    max_depth: int
    worst_lower: arb
    worst_interval: tuple[Fraction, Fraction]


def interval_ball(a: Fraction, b: Fraction) -> arb:
    """Return an Arb ball containing the exact rational interval [a,b]."""
    if b < a:
        raise ValueError("reversed interval")
    mid = (a + b) / 2
    rad = (b - a) / 2
    return arb(str(mid), str(rad))


def lower_is_positive(x: arb) -> bool:
    return bool(x.lower() > ZERO)


def fmt(x: arb, digits: int = 16) -> str:
    return x.str(digits, radius=False)


def certify_positive(
    name: str,
    functions: Sequence[Callable[[arb], arb]],
    a: Fraction,
    b: Fraction,
    *,
    max_depth: int = 40,
) -> list[Certificate]:
    """Certify that every function is positive on [a,b]."""
    stack: list[tuple[Fraction, Fraction, int]] = [(a, b, 0)]
    boxes = 0
    depth_used = 0
    worst: list[arb | None] = [None] * len(functions)
    worst_interval: list[tuple[Fraction, Fraction] | None] = [None] * len(functions)

    while stack:
        lo, hi, depth = stack.pop()
        p = interval_ball(lo, hi)
        try:
            values = [f(p) for f in functions]
        except (ValueError, ZeroDivisionError):
            values = []

        if values and all(lower_is_positive(v) for v in values):
            boxes += 1
            depth_used = max(depth_used, depth)
            for j, value in enumerate(values):
                lb = value.lower()
                if worst[j] is None or lb < worst[j]:
                    worst[j] = lb
                    worst_interval[j] = (lo, hi)
            continue

        if depth >= max_depth:
            details = ", ".join(v.str(12) for v in values) if values else "evaluation failed"
            raise RuntimeError(
                f"{name}: failed to certify [{lo},{hi}] at depth {depth}: {details}"
            )

        mid = (lo + hi) / 2
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))

    out: list[Certificate] = []
    for j in range(len(functions)):
        assert worst[j] is not None and worst_interval[j] is not None
        out.append(Certificate(name, boxes, depth_used, worst[j], worst_interval[j]))
    return out


def s_a(alpha: arb) -> arb:
    """sin(pi alpha/2)."""
    return (alpha * HALF).sin_pi()


def c3(alpha: arb) -> arb:
    return (
        arb.pi()
        * (ONE + alpha / 2).gamma()
        / ((ONE - alpha / 2).gamma() * ((ONE + alpha) / 2).gamma() ** 2)
    )


def c6(alpha: arb) -> arb:
    s = s_a(alpha)
    return 2 + (5 * s + s ** 2) / 4


def c7(alpha: arb) -> arb:
    return (2 * alpha).gamma() / (3 * alpha.gamma())


def c8(alpha: arb) -> arb:
    s = s_a(alpha)
    return ((ONE - 2 / c6(alpha)) / (c7(alpha) * s ** 2)).sqrt()


def g1_series(alpha: arb, u: arb, N: int) -> arb:
    """The odd-degree Taylor lower bound g_{1,N}(u) for g_1(u) = u^2 f_2(u)."""
    if N % 2 == 0:
        raise ValueError("g_{1,N} is a lower bound for g_1 only for odd N")
    total = arb(0)
    for k in range(N + 1):
        total += ((-u) ** k) * ((k + 2) * alpha / 2).gamma() / factorial(k)
    return alpha / 2 * u ** 2 * total


def threshold(alpha: arb) -> arb:
    s = s_a(alpha)
    q = (ONE + s ** 2).sqrt()
    return 2 / (c3(alpha) * (q + s)) ** 2


def localization(alpha: arb) -> arb:
    return g1_series(alpha, c8(alpha), 11) - threshold(alpha)


def report(label: str, certs: Sequence[Certificate]) -> None:
    print(label)
    for j, cert in enumerate(certs, start=1):
        lo, hi = cert.worst_interval
        print(
            f"  inequality {j}: PASS; terminal boxes={cert.boxes}; "
            f"max depth={cert.max_depth}; "
            f"smallest accepted lower endpoint={fmt(cert.worst_lower)} "
            f"on [{lo},{hi}]"
        )


def main() -> None:
    print(f"python-flint Arb verification at {ctx.prec} bits")
    report(
        "localization certificate on [1/6,1]",
        certify_positive("localization certificate", (localization,), Q(1, 6), Q(1, 1)),
    )
    print("CERTIFICATE PASSED")


if __name__ == "__main__":
    main()
