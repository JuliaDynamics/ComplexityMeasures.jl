# from before https://github.com/JuliaDynamics/ComplexityMeasures.jl/pull/239
function entropy(e::EntropyDefinition, est::DiffEntropyEst, x)
    if e isa Shannon
        return entropy(est, x)
    else
        error("only shannon entropy supports this deprecated interface")
    end
end

@deprecate ComplexityMeasure ComplexityEstimator

# From before 2.0:
@deprecate TimeScaleMODWT WaveletOverlap
function probabilities(x::Vector_or_Dataset, ε::Union{Real, Vector{<:Real}})
    @warn """
    `probabilities(x::Vector_or_Dataset, ε::Real)`
    is deprecated, use `probabilities(ValueHistogram(ε), x)`.
    """
    probabilities(ValueHistogram(ε), x)
end

function probabilities(x, est::ProbabilitiesEstimator)
    @warn """
    `probabilities(x, est::ProbabilitiesEstimator)`
    is deprecated, use `probabilities(est::ProbabilitiesEstimator, x) instead`.
    """
    return probabilities(est, x)
end

export genentropy, permentropy

function permentropy(x; τ = 1, m = 3, base = MathConstants.e)
    @warn """
    `permentropy(x; τ, m, base)` is deprecated.
    Use instead: `entropy_permutation(x; τ, m, base)`, or even better, use the
    direct syntax discussed in the docstring of `entropy_permutation`.
    """
    return entropy_permutation(x; τ, m, base)
end

function genentropy(probs::Probabilities; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(probs::Probabilities; q, base)` deprecated.
    Use instead: `entropy(Renyi(q, base), probs)`.
    """
    return entropy(Renyi(q, base), probs)
end

function genentropy(x::Array_or_Dataset, ε::Real; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_Dataset, ε::Real; q, base)` is deprecated.
    Use instead: `entropy(Renyi(q, base), ValueHistogram(ε), x)`.
    """
    return entropy(Renyi(q, base), ValueHistogram(ε), x)
end

function genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q, base)` is deprecated.
    Use instead: `entropy(Renyi(q, base), est, x)`.
    """
    return entropy(Renyi(q, base), x, est)
end
