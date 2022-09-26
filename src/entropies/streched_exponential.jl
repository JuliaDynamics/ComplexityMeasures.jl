using SpecialFunctions

export entropy_stretched_exponential
export maxentropy_stretched_exponential

function _se(pᵢ, η, base)
    x = (η + 1)/η
    # Note gamma_inc(a, b) returns (lower, upper) incomplete gamma functions,
    # scaled by 1/Γ(b), so we multiply by gamma(x) to obtain the non-normalized
    # integral used in Anteneodo & Plastino (1999). See
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.gamma_inc
    Γx = gamma(x)

    return gamma_inc(x, -log(base, pᵢ))[2] * Γx - pᵢ * Γx
end

"""
    entropy_stretched_exponential(p::Probabilities; η = 2.0, base = MathConstants.e)

Compute the stretched exponential entropy (Anteneodo & Plastino, 1999[^Anteneodo1999]),

```math
S_{\\eta}(p) = \\sum_{i = 1}^N
\\Gamma \\left( \\dfrac{\\eta + 1}{\\eta}, - \\log_{base}(p_i) \\right) -
p_i \\Gamma \\left( \\dfrac{\\eta + 1}{\\eta} \\right),
```

where ``\\eta \\geq 0``, ``\\Gamma(\\cdot, \\cdot)`` is the upper incomplete Gamma
function, and ``\\Gamma(\\cdot) = \\Gamma(\\cdot, 0)`` is the Gamma function. Reduces to
Shannon entropy for `η = 1.0`.

[^Anteneodo1999]: Anteneodo, C., & Plastino, A. R. (1999). Maximum entropy approach to
    stretched exponential probability distributions. Journal of Physics A: Mathematical
    and General, 32(7), 1089.
"""
function entropy_stretched_exponential(prob::Probabilities; η = 2.0, base = MathConstants.e)
    probs = Iterators.filter(!iszero, prob.p)
    return sum(_se(pᵢ, η, base) for pᵢ in probs)
end

function entropy_stretched_exponential(x, est; kwargs...)
    p = probabilities(x, est)
    entropy_streched_exponential(p; kwargs...)
end

"""
    maxentropy_stretched_exponential(N::Int; η = 2.0, base = MathConstants.e)

Convenience function that computes the maximum value of the streched exponental entropy
with parameter `η`, i.e.

```math
S_{\\eta}^{uniform}(N) = N \\Gamma \\left( \\dfrac{\\eta + 1}{\\eta}, -log_{base}(N)\\right) -
\\Gamma \\left( \\dfrac{\\eta + 1}{\\eta} \\right),
```

which occurs for a uniform distribution, and is useful for normalization when `N` is known.

See also [`entropy_stretched_exponential`](@ref).
"""
function maxentropy_stretched_exponential(N::Int; η = 2.0, base = MathConstants.e)
    x = (η + 1)/η
    Γx = gamma(x)
    # We need the scaled (se comment for `entropy_streched_exponential`) *upper* incomplete
    # gamma function, which is the second entry in the tuple returned from gamma_inc
    N * gamma_inc(x, log(base, N))[2] * Γx - Γx
end
