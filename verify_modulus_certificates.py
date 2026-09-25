#!/usr/bin/env python3
"""Rigorous Arb verification of the two scalar modulus comparisons.

Requires python-flint 0.9.0.  This script certifies the only numerical
inequalities used after the analytic strong-convexity estimate for C.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence

from flint import arb, ctx

ctx.prec = 256

Q = Fraction
ZERO = arb(0)
ONE = arb(1)
HALF = arb("1/2")


@dataclass
class Result:
    name: str
    boxes: int
    max_depth: int
    worst_lower: arb
    worst_interval: tuple[Fraction, Fraction]


def interval_ball(a: Fraction, b: Fraction) -> arb:
    mid = (a + b) / 2
    rad = (b - a) / 2
    return arb(str(mid), str(rad))


def D_p(p: arb) -> arb:
    return arb.pi() * (ONE + p).gamma() / ((ONE - p).gamma() * (p + HALF).gamma() ** 2)


def T_p(p: arb) -> arb:
    s = p.sin_pi()
    c = p.cos_pi()
    Qp = ((ONE - s ** 4).sqrt() - s * c) / D_p(p)
    return (ONE / Qp + c) / 2


def curvature(p: arb) -> arb:
    """The exact strong-convexity constant m_p for p < 1/2."""
    base = (ONE - p) * (ONE - 2 * p) / (2 * p ** 3)
    return ((2 * p) * base.log()).exp() / (ONE - 2 * p)


def low_denominator(p: arb) -> arb:
    s = p.sin_pi()
    return ONE - 4 * s ** 2 / curvature(p)


def low_margin(p: arb) -> arb:
    return T_p(p) - ONE / low_denominator(p)


def high_threshold(p: arb) -> arb:
    """Nonsingular form of sqrt(1-s^4) T_p."""
    s = p.sin_pi()
    c = p.cos_pi()
    root = (ONE + s ** 2).sqrt()
    return root * (D_p(p) * (root + s) + c ** 2) / 2


def high_margin(p: arb) -> arb:
    s = p.sin_pi()
    return high_threshold(p) - ONE / (ONE - s ** 2 / 2)


def certify(
    name: str,
    funcs: Sequence[Callable[[arb], arb]],
    a: Fraction,
    b: Fraction,
    *,
    max_depth: int = 40,
) -> list[Result]:
    stack = [(a, b, 0)]
    boxes = 0
    depth_used = 0
    worst: list[arb | None] = [None] * len(funcs)
    worst_box: list[tuple[Fraction, Fraction] | None] = [None] * len(funcs)

    while stack:
        lo, hi, depth = stack.pop()
        p = interval_ball(lo, hi)
        try:
            values = [f(p) for f in funcs]
        except (ValueError, ZeroDivisionError):
            values = []

        if values and all(bool(v.lower() > ZERO) for v in values):
            boxes += 1
            depth_used = max(depth_used, depth)
            for j, value in enumerate(values):
                lb = value.lower()
                if worst[j] is None or lb < worst[j]:
                    worst[j] = lb
                    worst_box[j] = (lo, hi)
            continue

        if depth >= max_depth:
            detail = ", ".join(v.str(12) for v in values) if values else "evaluation failed"
            raise RuntimeError(f"{name}: failed on [{lo},{hi}]: {detail}")

        mid = (lo + hi) / 2
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))

    results = []
    for j in range(len(funcs)):
        assert worst[j] is not None and worst_box[j] is not None
        results.append(Result(name, boxes, depth_used, worst[j], worst_box[j]))
    return results


def print_results(label: str, results: list[Result]) -> None:
    print(label)
    for j, result in enumerate(results, start=1):
        lo, hi = result.worst_interval
        print(
            f"  inequality {j}: PASS; terminal boxes={result.boxes}; "
            f"max depth={result.max_depth}; "
            f"worst lower bound={result.worst_lower.str(16, radius=False)}; "
            f"worst box=[{lo},{hi}]"
        )


def main() -> None:
    print(f"python-flint Arb verification at {ctx.prec} bits")
    print_results(
        "low-range modulus comparison",
        certify(
            "low-range modulus comparison",
            (low_denominator, low_margin),
            Q(4, 25),
            Q(1, 4),
        ),
    )
    print_results(
        "high-range phase-free comparison",
        certify(
            "high-range phase-free comparison",
            (high_margin,),
            Q(1, 4),
            Q(1, 2),
        ),
    )
    print("ALL MODULUS CERTIFICATES PASSED")


if __name__ == "__main__":
    main()
