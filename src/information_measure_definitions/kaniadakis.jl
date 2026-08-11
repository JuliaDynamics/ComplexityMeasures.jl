export Kaniadakis

"""
    Kaniadakis <: InformationMeasure
    Kaniadakis(; κ = 1.0, base = 2.0)

The Kaniadakis entropy [Tsallis2009](@cite), used with [`information`](@ref)
to compute

```math
H_K(p) = \\sum_{i=1}^N p_i \\log_\\kappa{\\left( 1/p_i \\right)}
       = -\\sum_{i=1}^N p_i \\log_\\kappa{(p_i)},
```
```math
\\log_\\kappa{(x)} = \\dfrac{x^\\kappa - x^{-\\kappa}}{2\\kappa \\ln{(base)}},
```

with the ``\\kappa``-logarithm at the given `base`. If ``\\kappa = 0``, the ordinary
logarithm to the given `base` is used, which is the continuous limit of the expression
above, and 0 probabilities are skipped.

`base` sets the (dimensionless) scale of the returned value. Use `base = MathConstants.e`
to recover the ``\\kappa``-logarithm in its usual published form
``(x^\\kappa - x^{-\\kappa})/(2\\kappa)``.
See [Units and the `base` keyword](@ref units_and_base).

The [`self_information`](@ref) is
``I_\\kappa(p_i) = \\log_\\kappa{(1/p_i)} = -\\log_\\kappa{(p_i)}``.
"""
Base.@kwdef struct Kaniadakis{K<:Real,B<:Real} <: Entropy
    κ::K = 1.0
    base::B = 2.0
end

function information(e::Kaniadakis, probs::Probabilities)
    κ = e.κ
    return - sum(pᵢ * logκ(e.base, pᵢ, κ) for pᵢ in probs)
end

function logκ(base, x, κ)
    if x == 0
        return 0.0
    end
    if κ == 0
        return log(base, x)
    else
        # The κ-logarithm is taken to the given `base`, i.e. scaled by 1/log(base), so
        # that the κ → 0 limit is continuous with the branch above.
        return (x^κ - x^(-κ)) / (2 * κ * log(base))
    end
end

function information_maximum(e::Kaniadakis, L::Int)
    throw(ErrorException("information_maximum not implemeted for Kaniadakis entropy yet"))
end

function self_information(e::Kaniadakis, pᵢ, N=nothing)
    return -logκ(e.base, pᵢ, e.κ)
end
