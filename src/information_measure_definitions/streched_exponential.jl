using SpecialFunctions: gamma, gamma_inc

export StretchedExponential

"""
    StretchedExponential <: InformationMeasure
    StretchedExponential(; η = 2.0, base = 2)

The stretched exponential, or Anteneodo-Plastino, entropy (Anteneodo &
Plastino, 1999[Anteneodo1999](@cite)), used with [`information`](@ref) to compute

```math
S_{\\eta}(p) = \\sum_{i = 1}^N
\\Gamma \\left( \\dfrac{\\eta + 1}{\\eta}, - \\log_{base}(p_i) \\right) -
p_i \\Gamma \\left( \\dfrac{\\eta + 1}{\\eta} \\right),
```

where ``\\eta \\geq 0``, ``\\Gamma(\\cdot, \\cdot)`` is the upper incomplete Gamma
function, and ``\\Gamma(\\cdot) = \\Gamma(\\cdot, 0)`` is the Gamma function. Reduces to
[Shannon](@ref) entropy for `η = 1.0`.

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
    x = (η + 1)/η
    # Note gamma_inc(a, b) returns (lower, upper) incomplete gamma functions,
    # scaled by 1/Γ(b), so we multiply by gamma(x) to obtain the non-normalized
    # integral used in Anteneodo & Plastino (1999). See
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.gamma_inc
    Γx = gamma(x)

    return gamma_inc(x, -log(base, pᵢ))[2] * Γx - pᵢ * Γx
end


function information(e::StretchedExponential, prob::Probabilities)
    probs = Iterators.filter(!iszero, prob.p)
    return sum(stretched_exponential(pᵢ, e.η, e.base) for pᵢ in probs)
end

function information_maximum(e::StretchedExponential, L::Int)
    x = (e.η + 1)/e.η
    Γx = gamma(x)
    # We need the scaled  *upper* incomplete gamma function, which is the second
    # entry in the tuple returned from `gamma_inc`.
    L * gamma_inc(x, log(e.base, L))[2] * Γx - Γx
end
