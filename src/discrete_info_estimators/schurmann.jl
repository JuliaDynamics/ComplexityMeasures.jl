export Schürmann

using SpecialFunctions: digamma
using QuadGK

"""
    Schürmann <: DiscreteInfoEstimator
    Schürmann(definition::Shannon; a = 1.0)

The `Schürmann` estimator computes the [`Shannon`](@ref) discrete
[`information`](@ref) with the bias-corrected estimator
given in Schürmann (2004)[^Schürmann2004].

See detailed description for [`GeneralizedSchürmann`](@ref) for details.

[^Schürmann2004]:
    Schürmann, T. (2004). Bias analysis in entropy estimation. Journal of Physics A:
    Mathematical and General, 37(27), L295.
"""
Base.@kwdef struct Schürmann{I <: InformationMeasure, A} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
    a::A = 1.0
end
function Schürmann(definition::I; a::A = 1.0) where {I, A}
    a > 0 || throw(ArgumentError("a must be strict positive. Got $a."))
    return Schürmann(; definition, a)
end

function information(hest::Schürmann{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; definition, a) = hest
    # We should be using `N = length(x)`, but since some probabilities estimators
    # return pseudo counts, we need to consider those instead of counting actual
    # observations.
    freqs = frequencies(pest, x)
    N = sum(freqs)

    h = digamma(N) - 1/N * sum(nᵢ * Sₙ(a, nᵢ) for nᵢ in freqs)

    # The Schürmann estimate of `h` is based on the natural logarithm, so we must convert
    # to the desired base.
    return convert_logunit(h, MathConstants.e, definition.base)
end

function Sₙ(a, n)
    if n == 0
        return 0.0
    end
    # Integrate `f_schurmann` from `lb` to `ub`. We only need the value of the integral
    # (not the error), so index first element returned from `quadgk`
    lb = 0.0
    ub = 1/a - 1 # Assumes a > 0, which is handled in the constructor to `Schürmann`.
    if lb == ub
        return digamma(n)
    end
    integ = first(quadgk(x -> f_schurmann(x, n), lb, ub))

    return digamma(n) + (-1)^n * integ
end

f_schurmann(t, n) = t^(n - 1) / (1 + t)
