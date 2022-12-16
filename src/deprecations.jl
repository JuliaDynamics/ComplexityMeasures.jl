@deprecate TimeScaleMODWT WaveletOverlap

function probabilities(x::Vector_or_Dataset, ε::Union{Real, Vector{<:Real}})
    @warn """
    `probabilities(x::Vector_or_Dataset, ε::Real)`
    is deprecated, use `probabilities(x, ValueHistogram(ε))`.
    """
    probabilities(x, ValueHistogram(ε))
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
    `permentropy(x; τ, m, base)` is a deprecated function in Entropies.jl v2.0.
    Use instead: `entropy_permutation(x; τ, m, base)`, or even better, use the
    direct syntax discussed in the docstring of `entropy_permutation`.
    """
    return entropy_permutation(x; τ, m, base)
end

function genentropy(probs::Probabilities; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(probs::Probabilities; q, base)` is a deprecated function in
    Entropies.jl v2.0. Use instead: `entropy(Renyi(q, base), probs)`.
    """
    return entropy(Renyi(q, base), probs)
end

function genentropy(x::Array_or_Dataset, ε::Real; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_Dataset, ε::Real; q, base)` is a deprecated function in
    Entropies.jl v2.0. Use instead: `entropy(Renyi(q, base), x, ValueHistogram(ε))`,
    or `entropy(x, ValueHistogram(ε))` if `q, base` have their default values.
    """
    return entropy(Renyi(q, base), x, ValueHistogram(ε))
end

function genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q, base)` is a deprecated function in
    Entropies.jl v2.0. Use instead: `entropy(Renyi(q, base), x, est)`,
    or `entropy(x, est)` if `q, base` have their default values.
    """
    return entropy(Renyi(q, base), x, est)
end
