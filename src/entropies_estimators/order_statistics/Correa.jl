export Correa

"""
    Correa <: DiffEntropyEst
    Correa(; m::Int = 1)

The `Correa` estimator computes the [`Shannon`](@ref) differential [`entropy`](@ref) of `x`
(a multi-dimensional [`Dataset`](@ref)) using the method from Correa (1995)[^Correa1995].

The `Correa` estimator belongs to a class of differential entropy estimators based
on [order statistics](https://en.wikipedia.org/wiki/Order_statistic). It only works for
*timeseries* input.

## Description

Assume we have samples ``\\bar{X} = \\{x_1, x_2, \\ldots, x_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R} \\to \\mathbb{R}``. `Correa` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

However, instead of estimating the above integral directly, `Correa` makes use of the
equivalent integral, where ``F`` is the distribution function for ``X``,

```math
H(X) = \\int_0^1 \\log \\left(\\dfrac{d}{dp}F^{-1}(p) \\right) dp
```

This integral is approximated by first computing the
[order statistics](https://en.wikipedia.org/wiki/Order_statistic) of ``\\bar{X}``
(the input timeseries), i.e. ``x_{(1)} \\leq x_{(2)} \\leq \\cdots \\leq x_{(n)}``,
ensuring that end points are included. The `Correa` estimate of [`Shannon`](@ref)
differential entropy is then

```math
H_C(\\bar{X}, m, n) =
\\dfrac{1}{n} \\sum_{i = 1}^n \\log
\\left[ \\dfrac{ \\sum_{j=i-m}^{i+m}(\\bar{X}_{(j)} -
\\tilde{X}_{(i)})(j - i)}{n \\sum_{j=i-m}^{i+m} (\\bar{X}_{(j)} - \\tilde{X}_{(i)})^2}
\\right],
```

where

```math
\\tilde{X}_{(i)} = \\dfrac{1}{2m + 1} \\sum_{j = i - m}^{i + m} X_{(j)}.
```

[^Correa1995]:
    Correa, J. C. (1995). A new estimator of entropy. Communications in Statistics-Theory
    and Methods, 24(10), 2439-2449.

See also: [`entropy`](@ref), [`AlizadehArghami`](@ref), [`Ebrahimi`](@ref),
[`Vasicek`](@ref), [`DifferentialEntropyEstimator`](@ref).
"""
@Base.kwdef struct Correa{I<:Integer} <: DiffEntropyEst
    m::I = 1
end

function entropy(e::Shannon, est::Correa, x::AbstractVector{T}) where T
    (; m) = est
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
        HCₘₙ += log(num / den)
    end
    return (-HCₘₙ / n) / log(e.base, ℯ)
end

function local_scaled_mean(ex, i::Int, m::Int, n::Int = length(x))
    x̄ = 0.0
    for j in (i - m):(i + m)
        x̄ += ith_order_statistic(ex, j, n) # ex[j] would cause out-of-bounds errors
    end

    return x̄ / (2m + 1)
end
