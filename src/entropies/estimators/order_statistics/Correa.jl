export Correa

"""
    Correa <: EntropyEstimator
    Correa(; m::Int = 1, base = 2)

The `Correa` estimator computes the [`Shannon`](@ref) [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) to the given `base` using the method from
Correa (1995)[^Correa1995].

## Description

The Correa entropy estimator first computes the order statistics like [`Vasicek`](@ref),
ensuring that edge points are included, then estimates entropy as

```math
H_C(m, n) =
\\dfrac{1}{n} \\sum_{i = 1}^n \\log
\\left[ \\dfrac{ \\sum_{j=i-m}^{i+m}(X_{(j)} - \\bar{X}_{(i)})(j - i)}{n \\sum_{j=i-m}^{i+m} (X_{(j)} - \\bar{X}_{(i)})^2} \\right],
```

where

```math
\\bar{X}_{(i)} = \\dfrac{1}{2m + 1} \\sum_{j = i - m}^{i + m} X_{(j)}.
```

[^Correa1995]:
    Correa, J. C. (1995). A new estimator of entropy. Communications in Statistics-Theory
    and Methods, 24(10), 2439-2449.
"""
@Base.kwdef struct Correa{I<:Integer, B} <: EntropyEstimator
    m::I = 1
    base::B = 2
end

function local_scaled_mean(ex, i::Int, m::Int, n::Int = length(x))
    x̄ = 0.0
    for j in (i - m):(i + m)
        x̄ += ith_order_statistic(ex, j, n) # ex[j] would cause out-of-bounds errors
    end

    return x̄ / (2m + 1)
end

function entropy(e::Correa, x::AbstractVector{T}) where T
    (; m, base) = e
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))

    ex = sort(x)
    HCₘₙ = 0.0
    for i = 1:n
        x̄ᵢ = local_scaled_mean(ex, i, m, n)
        num = 0.0
        den = 0.0
        for j in (i - m):(i + m)
            xⱼ = ith_order_statistic(ex, j, n)
            num += (xⱼ - x̄ᵢ) * (j - i)
            den += (xⱼ - x̄ᵢ)^2
        end
        den *= n
        HCₘₙ += log(base, num / den)
    end
    return -HCₘₙ / n
end
