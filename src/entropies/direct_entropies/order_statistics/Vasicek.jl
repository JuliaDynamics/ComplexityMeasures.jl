export Vasicek

"""
    Vasicek <: IndirectEntropy
    Vasicek(; m::Int = 1, base = 2)

An indirect entropy estimator used in [`entropy`](@ref)`(Vasicek(), x)` to
estimate the Shannon entropy of the timeseries `x` to the given
`base` using the method from Vasicek (1976)[^Vasicek1976].

## Description

The Vasicek entropy estimator first computes the order statistics
``X_{(1)} \\leq X_{(2)} \\leq \\cdots \\leq X_{(n)}`` for a random sample of length ``n``,
i.e. the input timeseries. The entropy for the length-`n` sample is then estimated as

```math
H_V(m, n) =
\\dfrac{1}{n} \\sum_{i = 1}^n \\log \\left[ \\dfrac{n}{2m} (X_{(i+m)} - X_{(i-m)}) \\right]
```

## Usage

In practice, choice of `m` influences how fast the entropy convergence to the true value.
For small value of `m`, convergence is slow, so we recommend to scale `m` according to the
time series length `n` and use `m >= n/100` (this is just a heuristic based on the tests
written for this package).

[^Vasicek1976]:
    Vasicek, O. (1976). A test for normality based on sample entropy. Journal of the Royal
    Statistical Society: Series B (Methodological), 38(1), 54-59.
"""
@Base.kwdef struct Vasicek{I<:Integer, B} <: IndirectEntropy
    m::I = 1
    base::B = 2
end

"""
    ith_order_statistic(ex, i::Int, n::Int = length(x))

Return the i-th order statistic from the order statistics `ex`, requiring that
`Xᵢ = X₁` if `i < 1` and `Xᵢ = Xₙ` if `i > n`.
"""
function ith_order_statistic(ex, i::Int, n::Int = length(x))
    if i < 1
        return ex[1]
    elseif i > n
        return ex[n]
    else
        return ex[i]
    end
end

function entropy(e::Vasicek, x::AbstractVector{T}) where T
    (; m, base) = e
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))

    ex = sort(x)
    HVₘₙ = 0.0
    f = n / (2m)
    for i = 1:n
        dnext = ith_order_statistic(ex, i + m, n)
        dprev = ith_order_statistic(ex, i - m, n)
        HVₘₙ += log(base, f * (dnext - dprev))
    end
    return HVₘₙ / n
end
