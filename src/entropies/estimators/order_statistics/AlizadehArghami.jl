export AlizadehArghami

"""
    AlizadehArghami <: EntropyEstimator
    AlizadehArghami(; m::Int = 1, base = 2)

The `AlizadehArghami`estimator computes the [`Shannon`](@ref) [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) to the given `base` using the method from Alizadeh & Arghami
(2010)[^Alizadeh2010].

## Description

The estimator first computes the
[order statistics](https://en.wikipedia.org/wiki/Order_statistic)
``X_{(1)} \\leq X_{(2)} \\leq \\cdots \\leq X_{(n)}`` for a random sample of length ``n``,
i.e. the input timeseries. The entropy for the length-`n` sample is then estimated as
the [`Vasicek`](@ref) entropy estimate, plus a correction factor

```math
H_{A}(m, n) = H_{V}(m, n) + \\dfrac{2}{n}\\left(m \\log_{base}(2) \\right).
```

See also: [`entropy`](@ref).

[^Alizadeh2010]:
    Alizadeh, N. H., & Arghami, N. R. (2010). A new estimator of entropy.
    Journal of the Iranian Statistical Society (JIRSS).
"""
@Base.kwdef struct AlizadehArghami{I<:Integer, B} <: EntropyEstimator
    m::I = 1
    base::B = 2
end

function entropy(e::Renyi, est::AlizadehArghami, x::AbstractVector{T}) where T
    e.q == 1 || throw(ArgumentError(
        "Renyi entropy with q = $(e.q) not implemented for $(typeof(est)) estimator"
    ))

    (; m, base) = est
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))
    return entropy(Vasicek(; m, base), x) + (2 / n)*(m * log(base, 2))
end
