export AlizadehArghami

"""
    AlizadehArghami <: EntropyEstimator
    AlizadehArghami(; m::Int = 1)

The `AlizadehArghami`estimator computes the [`Shannon`](@ref) differential
[`entropy`](@ref) of `x` (a multi-dimensional `Dataset`) using the
method from Alizadeh & Arghami (2010)[^Alizadeh2010].

The `AlizadehArghami` estimator belongs to a class of differential entropy estimators based
on [order statistics](https://en.wikipedia.org/wiki/Order_statistic). It only works for
*timeseries* input.

## Description

Assume we have samples ``\\bar{X} = \\{x_1, x_2, \\ldots, x_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R} \\to \\mathbb{R}``. `AlizadehArghami` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

However, instead of estimating the above integral directly, it makes use of the equivalent
integral, where ``F`` is the distribution function for ``X``:

```math
H(X) = \\int_0^1 \\log \\left(\\dfrac{d}{dp}F^{-1}(p) \\right) dp.
```

This integral is approximated by first computing the
[order statistics](https://en.wikipedia.org/wiki/Order_statistic) of ``\\bar{X}``
(the input timeseries), i.e. ``x_{(1)} \\leq x_{(2)} \\leq \\cdots \\leq x_{(n)}``.
The `AlizadehArghami` [`Shannon`](@ref) differential entropy estimate is then the
the [`Vasicek`](@ref) estimate ``\\hat{H}_{V}(\\bar{X}, m, n)``, plus a correction factor

```math
\\hat{H}_{A}(\\bar{X}, m, n) = \\hat{H}_{V}(\\bar{X}, m, n) +
\\dfrac{2}{n}\\left(m \\log(2) \\right).
```

[^Alizadeh2010]:
    Alizadeh, N. H., & Arghami, N. R. (2010). A new estimator of entropy.
    Journal of the Iranian Statistical Society (JIRSS).

See also: [`entropy`](@ref), [`Correa`](@ref), [`Ebrahimi`](@ref),
[`Vasicek`](@ref), [`EntropyEstimator`](@ref).
"""
@Base.kwdef struct AlizadehArghami{I<:Integer} <: EntropyEstimator
    m::I = 1
end

function entropy(e::Renyi, est::AlizadehArghami, x::AbstractVector{T}) where T
    e.q == 1 || throw(ArgumentError(
        "Renyi entropy with q = $(e.q) not implemented for $(typeof(est)) estimator"
    ))

    (; m) = est
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))
    h = entropy(Renyi(base = ℯ, q = e.q), Vasicek(; m), x) + (2 / n)*(m * log(2))
    return h / log(e.base, ℯ)
end
