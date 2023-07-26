export Schürmann

using SpecialFunctions: digamma
using QuadGK

"""
    Schürmann <: DiscreteInfoEstimator
    Schürmann(measure::Shannon; a = 1.0)

The `Schürmann` estimator computes the [`Shannon`](@ref) discrete
[`information`](@ref) with the bias-corrected estimator
given in Schürmann (2004)[^Schürmann2004].

[^Schürmann2004]:
    Schürmann, T. (2004). Bias analysis in entropy estimation. Journal of Physics A:
    Mathematical and General, 37(27), L295.
"""
struct Schürmann{I <: InformationMeasure, A} <: DiscreteInfoEstimator{I}
    measure::I
    a::A
end

function Schürmann(measure = Shannon(); a = 1.0)
    a > 0 || throw(ArgumentError("a must be strict positive. Got $a."))
    return Schürmann(measure, a)
end

function information(hest::Schürmann{<:Shannon}, pest::ProbabilitiesEstimator, x)
    (; measure, a) = hest
    N = length(x)
    freqs = frequencies(pest, x)
    h = digamma(N) - 1/N * sum(nᵢ * Sₙ(a, nᵢ) for nᵢ in freqs)

    # The Schürmann estimate of `h` is based on the natural logarithm, so we must convert
    # to the desired base.
    return convert_logunit(h, MathConstants.e, measure.base)
end

function Sₙ(a, n)
    # Integrate `f_schurmann` from `lb` to `ub`. We only need the value of the integral
    # (not the error), so index first element returned from `quadgk`
    lb = 0.0
    ub = 1/a - 1 # Assumes a > 0, which is handled in the constructor to `Schürmann`.
    integ = first(quadgk(x -> f_schurmann(x, n), lb, ub))

    return digamma(n) + (-1)^n * integ
end

f_schurmann(t, n) = t^(n - 1) / (1 + t)
