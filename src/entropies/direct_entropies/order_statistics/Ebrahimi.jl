export Ebrahimi

"""
    Ebrahimi <: IndirectEntropy
    Ebrahimi(; m::Int = 1, base = 2)

An indirect entropy estimator used in [`entropy`](@ref)`(Ebrahimi(), x)` to
estimate the Shannon entropy of the timeseries `x` to the given
`base` using the method from Ebrahimi (1994)[^Ebrahimi1994].

## Description

The Ebrahimi entropy estimator first computes the order statistics
``X_{(1)} \\leq X_{(2)} \\leq \\cdots \\leq X_{(n)}`` for a random sample of length ``n``,
i.e. the input timeseries. The [`Shannon`](@ref) entropy is then estimated as

```math
H_{E}(m) =
\\dfrac{1}{n} \\sum_{i = 1}^n \\log \\left[ \\dfrac{n}{c_i m} (X_{(i+m)} - X_{(i-m)}) \\right],
```

where

```math
c_i =
\\begin{cases}
    1 + \\frac{i - 1}{m}, & 1 \\geq i \\geq m \\
    2,                    & m + 1 \\geq i \\geq n - m \\
    1 + \\frac{n - i}{m} & n - m + 1 \\geq i \\geq n
\\end{cases}.
```

[^Ebrahimi1994]:
    Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). Two measures of sample entropy.
    Statistics & Probability Letters, 20(3), 225-234.
"""
@Base.kwdef struct Ebrahimi{I<:Integer, B} <: IndirectEntropy
    m::I = 1
    base::B = 2
end

function ebrahimi_scaling_factor(i, m, n)
    if 1 ≤ i ≤ m
        return 1 + (i - 1) / m
    elseif m + 1 ≤ i ≤ n - m
        return 2
    else n - m + 1 ≤ i ≤ n
        return 1 + (n - i) / m
    end
end

function entropy(e::Ebrahimi, x::AbstractVector{T}) where T
    (; m, base) = e
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))

    ex = sort(x)
    HVₘₙ = 0.0
    for i = 1:n
        cᵢ = ebrahimi_scaling_factor(i, m, n)
        f = n / (cᵢ * m)
        dnext = ith_order_statistic(ex, i + m, n)
        dprev = ith_order_statistic(ex, i - m, n)
        HVₘₙ += log(base, f * (dnext - dprev))
    end
    return HVₘₙ / n
end
