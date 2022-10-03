using SpecialFunctions
import Base.maximum

export StretchedExponential

"""
    StretchedExponential <: Entropy
    StretchedExponential(η = 2.0, base = 2)

The stretched exponential, or Anteneodo-Plastino, entropy (Anteneodo &
Plastino, 1999[^Anteneodo1999]), used with [`entropy`](@ref) to compute

```math
S_{\\eta}(p) = \\sum_{i = 1}^N
\\Gamma \\left( \\dfrac{\\eta + 1}{\\eta}, - \\log_{base}(p_i) \\right) -
p_i \\Gamma \\left( \\dfrac{\\eta + 1}{\\eta} \\right),
```

where ``\\eta \\geq 0``, ``\\Gamma(\\cdot, \\cdot)`` is the upper incomplete Gamma
function, and ``\\Gamma(\\cdot) = \\Gamma(\\cdot, 0)`` is the Gamma function. Reduces to
[Shannon](@ref) entropy for `η = 1.0`.

[^Anteneodo1999]: Anteneodo, C., & Plastino, A. R. (1999). Maximum entropy approach to
    stretched exponential probability distributions. Journal of Physics A: Mathematical
    and General, 32(7), 1089.
"""
struct StretchedExponential{Q, B} <: Entropy
    η::Q
    base::B
end

function StretchedExponential(; η = 2, base = 2)
    η >= 0 || throw(ArgumentError("Need η ≥ 0. Got η=$(η)."))
    StretchedExponential(η, base)
end

function _se(pᵢ, η, base)
    x = (η + 1)/η
    # Note gamma_inc(a, b) returns (lower, upper) incomplete gamma functions,
    # scaled by 1/Γ(b), so we multiply by gamma(x) to obtain the non-normalized
    # integral used in Anteneodo & Plastino (1999). See
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.gamma_inc
    Γx = gamma(x)

    return gamma_inc(x, -log(base, pᵢ))[2] * Γx - pᵢ * Γx
end


function entropy(e::StretchedExponential, prob::Probabilities)
    probs = Iterators.filter(!iszero, prob.p)
    return sum(_se(pᵢ, e.η, e.base) for pᵢ in probs)
end

function entropy(e::StretchedExponential, x, est::ProbabilitiesEstimator; kwargs...)
    p = probabilities(x, est)
    entropy(e, p; kwargs...)
end

function maximum(e::StretchedExponential, L::Int)
    x = (e.η + 1)/e.η
    Γx = gamma(x)
    # We need the scaled  *upper* incomplete gamma function, which is the second
    # entry in the tuple returned from `gamma_inc`.
    L * gamma_inc(x, log(e.base, L))[2] * Γx - Γx
end
