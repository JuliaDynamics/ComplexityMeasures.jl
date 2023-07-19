export Kaniadakis

"""
    Kaniadakis <: InformationMeasureDefinition
    Kaniadakis(; κ = 1.0, base = 2.0)

The Kaniadakis entropy (Tsallis, 2009)[^Tsallis2009], used with [`information`](@ref) to
compute

```math
H_K(p) = -\\sum_{i=1}^N p_i f_\\kappa(p_i),
```
```math
f_\\kappa (x) = \\dfrac{x^\\kappa - x^{-\\kappa}}{2\\kappa},
```
where if ``\\kappa = 0``, regular logarithm to the given `base` is used, and
0 probabilities are skipped.

[^Tsallis2009]:
    Tsallis, C. (2009). Introduction to nonextensive statistical mechanics: approaching a
    complex world. Springer, 1(1), 2-1.
"""
Base.@kwdef struct Kaniadakis{K <: Real, B <: Real} <: InformationMeasureDefinition
    κ::K = 1.0
    base::B = 2.0
end

function information(e::Kaniadakis, probs::Probabilities)
    κ = e.κ
    return - sum(pᵢ * logκ(e.base, pᵢ, κ)  for pᵢ in probs)
end

function logκ(base, x, κ)
    if x == 0
        return 0.0
    end
    if κ == 0
        return log(base, x)
    else
        return (x^κ - x^(-κ)) / (2 * κ)
    end
end

function information_maximum(e::Kaniadakis, L::Int)
    throw(ErrorException("information_maximum not implemeted for Kaniadakis entropy yet"))
end
