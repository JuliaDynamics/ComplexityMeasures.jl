export Vasicek

"""
    Vasicek <: DiffEntropyEst
    Vasicek(; m::Int = 1)

The `Vasicek` estimator computes the [`Shannon`](@ref) differential [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) using the method from Vasicek (1976)[^Vasicek1976].

The `Vasicek` estimator belongs to a class of differential entropy estimators based
on [order statistics](https://en.wikipedia.org/wiki/Order_statistic), of which
Vasicek (1976) was the first. It only works for *timeseries* input.

## Description

Assume we have samples ``\\bar{X} = \\{x_1, x_2, \\ldots, x_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R} \\to \\mathbb{R}``. `Vasicek` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

However, instead of estimating the above integral directly, it makes use of the equivalent
integral, where ``F`` is the distribution function for ``X``,

```math
H(X) = \\int_0^1 \\log \\left(\\dfrac{d}{dp}F^{-1}(p) \\right) dp
```

This integral is approximated by first computing the
[order statistics](https://en.wikipedia.org/wiki/Order_statistic) of ``\\bar{X}``
(the input timeseries), i.e. ``x_{(1)} \\leq x_{(2)} \\leq \\cdots \\leq x_{(n)}``.
The `Vasicek` [`Shannon`](@ref) differential entropy estimate is then

```math
\\hat{H}_V(\\bar{X}, m) =
\\dfrac{1}{n}
\\sum_{i = 1}^n \\log \\left[ \\dfrac{n}{2m} (\\bar{X}_{(i+m)} - \\bar{X}_{(i-m)}) \\right]
```

## Usage

In practice, choice of `m` influences how fast the entropy converges to the true value.
For small value of `m`, convergence is slow, so we recommend to scale `m` according to the
time series length `n` and use `m >= n/100` (this is just a heuristic based on the tests
written for this package).

[^Vasicek1976]:
    Vasicek, O. (1976). A test for normality based on sample entropy. Journal of the Royal
    Statistical Society: Series B (Methodological), 38(1), 54-59.

See also: [`entropy`](@ref), [`Correa`](@ref), [`AlizadehArghami`](@ref),
[`Ebrahimi`](@ref), [`DifferentialEntropyEstimator`](@ref).
"""
@Base.kwdef struct Vasicek{I<:Integer} <: DiffEntropyEst
    m::I = 1
end

function entropy(e::Renyi, est::Vasicek, x::AbstractVector{T}) where T
    e.q == 1 || throw(ArgumentError(
        "Renyi entropy with q = $(e.q) not implemented for $(typeof(est)) estimator"
    ))
    (; m) = est
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))

    ex = sort(x)
    HVₘₙ = zero(T)
    f = n / (2m)
    for i = 1:n
        dnext = ith_order_statistic(ex, i + m, n)
        dprev = ith_order_statistic(ex, i - m, n)
        HVₘₙ += log(f * (dnext - dprev))
    end
    return (HVₘₙ / n) / log(e.base, ℯ)
end
