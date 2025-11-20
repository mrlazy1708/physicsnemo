from typing import Protocol
from torch import Tensor

import torch
import torch.nn as nn

import numpy as np
import functools

class Basis(Protocol):

    grid: Tensor
    quad: Tensor

    m: int
    f: Tensor

    def fn(self, xs: Tensor) -> Tensor:
        """
        Evaluate basis functions at given points.
        """

    def at(self, coef: Tensor, xs: Tensor) -> Tensor:
        """
        Evaluate basis expansion at given points.
        """
        return torch.tensordot(self.fn(xs), coef, dims=1)

    def modal(self, vals: Tensor) -> Tensor:
        """
        Convert nodal values to modal coefficients.
        """

    def nodal(self, coef: Tensor) -> Tensor:
        """
        Convert modal coefficients to nodal values.
        """

class Element(Basis):

    def __init__(self, base: Basis):
        """
        """

# ---------------------------------------------------------------------------- #
#                                   LEGENDRE                                   #
# ---------------------------------------------------------------------------- #

from numpy.polynomial import legendre

@functools.cache
class Legendre(nn.Module, Basis):

    """
    Shifted Legendre polynomials:
    - `(1 - x^2) Pn''(x) - 2 x Pn(x) + n (n + 1) Pn(x) = 0`
    - `Pn^~(x) = Pn(2 x - 1)`
    """

    def extra_repr(self) -> str:
        return f"m={self.m}"

    def __init__(self, m: int, endpoint: bool = False):
        """
        """
        super().__init__()
        self.m = m

        if endpoint: m -= 1
        c = (0, ) * m + (1, )
        dc = legendre.legder(c)

        x = legendre.legroots(dc if endpoint else c)
        y = legendre.legval(x, c if endpoint else dc)

        if endpoint:
            x = np.concatenate([[-1], x, [1]])
            y = np.concatenate([[1], y, [1]])

        w = 1 / y ** 2
        if endpoint: w /= m * (m + 1)
        else: w /= 1 - x ** 2

        self.register_buffer("grid", torch.tensor((1 + x) / 2, dtype=torch.float))
        self.register_buffer("quad", torch.tensor(w, dtype=torch.float))

        self.register_buffer("f", self.fn(self.grid))

    def fn(self, xs: Tensor) -> Tensor:
        """
        """
        P = torch.ones_like(xs), 2 * xs - 1

        for i in range(2, self.m):
            a, b = (i * 2 - 1) / i, (i - 1) / i
            P += a * P[-1] * P[1] - b * P[-2],

        return torch.stack(P, dim=-1)

# --------------------------------- TRANSFORM -------------------------------- #

    def modal(self, vals: Tensor) -> Tensor:
        """
        """
        norm = 2 * torch.arange(self.m, device=vals.device) + 1
        coef = self.f * norm * self.quad[:, None]
        return torch.tensordot(coef.T, vals, dims=1)

    def nodal(self, coef: Tensor) -> Tensor:
        """
        """
        return self.at(coef, self.grid)
