export AlizadehArghami

"""
AlizadehArghami <: IndirectEntropy
    AlizadehArghami(; m::Int = 1, base = 2)

An indirect entropy estimator used in [`entropy`](@ref)`(Alizadeh(), x)` to
estimate the Shannon entropy of the timeseries `x` to the given
`base` using the method from Alizadeh & Arghami (2010)[^Alizadeh2010].

## Description

The Alizadeh entropy estimator first computes the order statistics
``X_{(1)} \\leq X_{(2)} \\leq \\cdots \\leq X_{(n)}`` for a random sample of length ``n``,
i.e. the input timeseries. The entropy for the length-`n` sample is then estimated as
the [`Vasicek`](@ref) entropy estimate, plus a correction factor

```math
H_{A}(m, n) = H_{V}(m, n) + \\dfrac{2}{n}\\left(m \\log_{base}(2) \\right).
```

[^Alizadeh2010]:
    Alizadeh, N. H., & Arghami, N. R. (2010). A new estimator of entropy.
    Journal of the Iranian Statistical Society (JIRSS).
"""
@Base.kwdef struct AlizadehArghami{I<:Integer, B} <: IndirectEntropy
    m::I = 1
    base::B = 2
end

function entropy(e::AlizadehArghami, x::AbstractVector{T}) where T
    (; m, base) = e
    n = length(x)
    m < floor(Int, n / 2) || throw(ArgumentError("Need m < length(x)/2."))
    return entropy(Vasicek(; m, base), x) + (2 / n)*(m * log(base, 2))
end
