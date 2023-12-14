export Schuermann

using SpecialFunctions: digamma
using QuadGK

"""
    Schuermann <: DiscreteInfoEstimator
    Schuermann(definition::Shannon; a = 1.0)

The `Schuermann` estimator is used with [`information`](@ref) to compute the
discrete [`Shannon`](@ref) entropy with the bias-corrected estimator
given in [Schurmann2004](@citet).

See detailed description for [`GeneralizedSchuermann`](@ref) for details.
"""
Base.@kwdef struct Schuermann{I <: InformationMeasure, A} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
    a::A = 1.0
end
function Schuermann(definition::I; a::A = 1.0) where {I, A}
    a > 0 || throw(ArgumentError("a must be strict positive. Got $a."))
    return Schuermann(; definition, a)
end
function information(hest::Schuermann{<:Shannon}, est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    (; definition, a) = hest

    # We should be using `N = length(x)`, but since some probabilities estimators
    # return pseudo counts, we need to consider those instead of counting actual
    # observations.
    cts = counts(o, x)
    N = sum(cts)

    h = digamma(N) - 1/N * sum(nᵢ * Sₙ(a, nᵢ) for nᵢ in cts)

    # The Schuermann estimate of `h` is based on the natural logarithm, so we must convert
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
    ub = 1/a - 1 # Assumes a > 0, which is handled in the constructor to `Schuermann`.
    if lb == ub
        return digamma(n)
    end
    integ = first(quadgk(x -> f_schurmann(x, n), lb, ub))

    return digamma(n) + (-1)^n * integ
end

f_schurmann(t, n) = t^(n - 1) / (1 + t)
