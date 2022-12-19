export Vasicek

"""
    Vasicek <: EntropyEstimator
    Vasicek(; m::Int = 1, base = 2)

The `Vasicek` estimator computes the [`Shannon`](@ref) differential [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) to the given `base` using the method from
Vasicek (1976)[^Vasicek1976].

## Description

The Vasicek entropy estimator first computes the
[order statistics](https://en.wikipedia.org/wiki/Order_statistic)
``X_{(1)} \\leq X_{(2)} \\leq \\cdots \\leq X_{(n)}`` for a random sample of length ``n``,
i.e. the input timeseries. The [`Shannon`](@ref) entropy is then estimated as

```math
H_V(m) =
\\dfrac{1}{n} \\sum_{i = 1}^n \\log \\left[ \\dfrac{n}{2m} (X_{(i+m)} - X_{(i-m)}) \\right]
```

## Usage

In practice, choice of `m` influences how fast the entropy converges to the true value.
For small value of `m`, convergence is slow, so we recommend to scale `m` according to the
time series length `n` and use `m >= n/100` (this is just a heuristic based on the tests
written for this package).

[^Vasicek1976]:
    Vasicek, O. (1976). A test for normality based on sample entropy. Journal of the Royal
    Statistical Society: Series B (Methodological), 38(1), 54-59.
"""
@Base.kwdef struct Vasicek{I<:Integer, B} <: EntropyEstimator
    m::I = 1
    base::B = 2
end

function entropy(e::Renyi, est::Vasicek, x::AbstractVector{T}) where T
    e.q == 1 || throw(ArgumentError(
        "Renyi entropy with q = $(e.q) not implemented for $(typeof(est)) estimator"
    ))
    (; m, base) = est
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))

    ex = sort(x)
    HVₘₙ = zero(T)
    f = n / (2m)
    for i = 1:n
        dnext = ith_order_statistic(ex, i + m, n)
        dprev = ith_order_statistic(ex, i - m, n)
        HVₘₙ += log(base, f * (dnext - dprev))
    end
    return HVₘₙ / n
end
