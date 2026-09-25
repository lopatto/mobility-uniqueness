#!/usr/bin/env python3
"""Rigorous Arb verification of the scalar certificates in uniqueness.tex.

Requires python-flint 0.9.0.  All interval endpoints are exact rational
numbers.  The program evaluates only the explicit finite formulas stated in
the manuscript and recursively bisects an interval whenever direct ball
evaluation does not yet prove strict positivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Callable, Iterable, Sequence

from flint import arb, ctx

ctx.prec = 256

Q = Fraction
ZERO = arb(0)
ONE = arb(1)
TWO = arb(2)
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


def bisect(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    m = (a + b) / 2
    return a, m, b


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

        lo, mid, hi = bisect(lo, hi)
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))

    out: list[Certificate] = []
    for j in range(len(functions)):
        assert worst[j] is not None and worst_interval[j] is not None
        out.append(Certificate(name, boxes, depth_used, worst[j], worst_interval[j]))
    return out


def s_p(p: arb) -> arb:
    return p.sin_pi()


def c_p(p: arb) -> arb:
    return p.cos_pi()


def D_p(p: arb) -> arb:
    return arb.pi() * (ONE + p).gamma() / ((ONE - p).gamma() * (p + HALF).gamma() ** 2)


def Q_p(p: arb) -> arb:
    s = s_p(p)
    c = c_p(p)
    return ((ONE - s ** 4).sqrt() - s * c) / D_p(p)


def T_p(p: arb) -> arb:
    return (ONE / Q_p(p) + c_p(p)) / 2


def K_p(p: arb) -> arb:
    return (4 * p).gamma() / (2 * p).gamma()


def B0_cutoff(p: arb) -> arb:
    return arb("6/5") * (ONE / p).log()


def beta_cutoff(p: arb) -> arb:
    return ONE - p * (ONE / p).log()


def B0_series(p: arb) -> arb:
    return arb("109/100") * (ONE / p).log() - p / 10


def L_cutoff(p: arb, B: arb) -> arb:
    beta = beta_cutoff(p)
    beta_to_inv_p = (beta.log() / p).exp()
    z = beta * B
    return (ONE - p) * c_p(p) * (-beta_to_inv_p).exp() * (ONE - (-z).exp() * (ONE + z))


def L_series(p: arb, B: arb, N: int) -> arb:
    total = arb(0)
    for k in range(N + 1):
        total += ((-B) ** k) * ((k + 2) * p).gamma() / factorial(k)
    return (ONE - p) * c_p(p) * p * B ** 2 * total


def J_series(p: arb, N: int) -> arb:
    total = arb(0)
    for k in range(N + 1):
        total += ((-1) ** k) * ((k + 1) * p).gamma() / factorial(k)
    return p * total


def low_cutoff_loc(p: arb) -> arb:
    B = B0_cutoff(p)
    return L_cutoff(p, B) - Q_p(p)


def low_cutoff_mod(p: arb) -> arb:
    B = B0_cutoff(p)
    return ONE - K_p(p) * B ** 2 * s_p(p) ** 2 / 2 - T_p(p).rsqrt()


def low_series_loc(p: arb) -> arb:
    B = B0_series(p)
    return L_series(p, B, 11) - Q_p(p)


def low_series_mod(p: arb) -> arb:
    B = B0_series(p)
    return ONE - K_p(p) * B ** 2 * s_p(p) ** 2 / 2 - T_p(p).rsqrt()


def radius_two(p: arb) -> arb:
    return L_series(p, arb(2), 7) - Q_p(p)


def c_star(p: arb) -> arb:
    return p.gamma() / (2 * p).gamma()


def a_star(p: arb) -> arb:
    return K_p(p) * s_p(p) ** 2 / 2


def f_tilt(p: arb, B: arb) -> arb:
    return B / (B + c_star(p)) * (ONE - a_star(p) * B ** 2)


def radius_one_functions(Bstar: Fraction) -> tuple[Callable[[arb], arb], ...]:
    B = arb(str(Bstar))
    return (
        lambda p: f_tilt(p, ONE) - Q_p(p),
        lambda p: f_tilt(p, B) - Q_p(p),
        lambda p: L_series(p, B, 7) - Q_p(p),
    )


def high_parameter(p: arb) -> arb:
    s = s_p(p)
    threshold = 2 * ((ONE + s ** 2).sqrt() - s) / D_p(p)
    return J_series(p, 5) - threshold


def report(label: str, certs: Iterable[Certificate]) -> None:
    print(label)
    for j, cert in enumerate(certs, start=1):
        lo, hi = cert.worst_interval
        print(
            f"  inequality {j}: PASS; terminal boxes={cert.boxes}; "
            f"max depth={cert.max_depth}; worst lower bound={fmt(cert.worst_lower)}; "
            f"worst box=[{lo},{hi}]"
        )


def main() -> None:
    jobs: list[tuple[str, Sequence[Callable[[arb], arb]], Fraction, Fraction]] = []

    cutoff_ranges = [
        (Q(1, 10000), Q(1, 1000)),
        (Q(1, 1000), Q(3, 1000)),
        (Q(3, 1000), Q(6, 1000)),
        (Q(6, 1000), Q(1, 100)),
        (Q(1, 100), Q(1, 50)),
        (Q(1, 50), Q(3, 100)),
    ]
    for lo, hi in cutoff_ranges:
        jobs.append((f"low cutoff [{lo},{hi}]", (low_cutoff_loc, low_cutoff_mod), lo, hi))

    series_ranges = [
        (Q(3, 100), Q(3, 50)),
        (Q(3, 50), Q(9, 100)),
        (Q(9, 100), Q(3, 25)),
        (Q(3, 25), Q(7, 50)),
        (Q(7, 50), Q(4, 25)),
    ]
    for lo, hi in series_ranges:
        jobs.append((f"low series [{lo},{hi}]", (low_series_loc, low_series_mod), lo, hi))

    jobs.append(("radius two", (radius_two,), Q(4, 25), Q(1, 4)))
    jobs.append(("radius one, first splice", radius_one_functions(Q(7, 4)), Q(1, 4), Q(29, 100)))
    jobs.append(("radius one, second splice", radius_one_functions(Q(419, 250)), Q(29, 100), Q(33, 100)))
    jobs.append(("high parameter", (high_parameter,), Q(33, 100), Q(1, 2)))

    print(f"python-flint Arb verification at {ctx.prec} bits")
    for name, funcs, lo, hi in jobs:
        report(name, certify_positive(name, funcs, lo, hi))
    print("ALL SCALAR CERTIFICATES PASSED")


if __name__ == "__main__":
    main()
