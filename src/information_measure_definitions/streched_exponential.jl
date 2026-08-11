using SpecialFunctions: gamma, gamma_inc

export StretchedExponential

"""
    StretchedExponential <: InformationMeasure
    StretchedExponential(; η = 2.0, base = 2)

The stretched exponential, or Anteneodo-Plastino, entropy [Anteneodo1999](@cite), used with
[`information`](@ref) to compute

```math
S_{\\eta}(p) = \\dfrac{1}{\\ln{(base)}} \\sum_{i = 1}^N \\left[
\\Gamma \\left( \\dfrac{\\eta + 1}{\\eta}, - \\ln(p_i) \\right) -
p_i \\Gamma \\left( \\dfrac{\\eta + 1}{\\eta} \\right) \\right],
```

where ``\\eta \\geq 0``, ``\\Gamma(\\cdot, \\cdot)`` is the upper incomplete Gamma
function, and ``\\Gamma(\\cdot) = \\Gamma(\\cdot, 0)`` is the Gamma function. Reduces to
[`Shannon`](@ref) entropy, at the same `base`, for `η = 1.0`.

`base` sets the (dimensionless) scale of the returned value: the expression published by
[Anteneodo1999](@citet) divided by ``\\ln{(base)}``. Use `base = MathConstants.e` to
recover that expression exactly.
See [Units and the `base` keyword](@ref units_and_base).

The maximum entropy for `StrechedExponential` is a rather complicated expression involving
incomplete Gamma functions (see source code).
"""
Base.@kwdef struct StretchedExponential{Q, B} <: Entropy
    η::Q = 2.0
    base::B = 2

    function StretchedExponential(η::Q, base::B) where {Q <: Real, B <: Real}
        η >= 0 || throw(ArgumentError("Need η ≥ 0. Got η=$(η)."))
        new{Q, B}(η, base)
    end
end

function stretched_exponential(pᵢ, η, base)
    x = (η + 1) / η
    # Note gamma_inc(a, b) returns (lower, upper) incomplete gamma functions,
    # scaled by 1/Γ(b), so we multiply by gamma(x) to obtain the non-normalized
    # integral used in Anteneodo & Plastino (1999). See
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.gamma_inc
    Γx = gamma(x)
    # `base` is applied as an overall 1/log(base) factor rather than inside the incomplete
    # gamma function. Unlike Shannon (where the base sits in the logarithm) or Tsallis and
    # Kaniadakis (where it sits in the deformed logarithm), there is no logarithm here to
    # absorb it: the natural logarithm below is fixed by the definition, and substituting
    # log(base, pᵢ) for it yields a different function rather than a change of units. For
    # base = 2 that substitution makes self_information negative for small pᵢ, and breaks
    # the reduction to Shannon at η = 1.
    return (gamma_inc(x, -log(pᵢ))[2] * Γx - pᵢ * Γx) / log(base)
end


function information(e::StretchedExponential, prob::Probabilities)
    probs = Iterators.filter(!iszero, prob.p)
    return sum(stretched_exponential(pᵢ, e.η, e.base) for pᵢ in probs)
end

function information_maximum(e::StretchedExponential, L::Int)
    x = (e.η + 1) / e.η
    Γx = gamma(x)
    # We need the scaled  *upper* incomplete gamma function, which is the second
    # entry in the tuple returned from `gamma_inc`.
    return (L * gamma_inc(x, log(L))[2] * Γx - Γx) / log(e.base)
end

function self_information(e::StretchedExponential, pᵢ, N=nothing)
    η, base = e.η, e.base
    Γ₁ = gamma((η + 1) / η, -log(pᵢ))
    Γ₂ = gamma((η + 1) / η)
    # NB! Filter for pᵢ != 0 before calling this method.
    return (Γ₁ / pᵢ - Γ₂) / log(base)
end