@deprecate TimeScaleMODWT WaveletOverlap
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
    Entropies.jl v2.0. Use instead: `entropy(Renyi(q, base), probabilities(x, ε))`,
    or `entropy(probabilities(x, ε))` if `q, base` have their default values.
    """
    return entropy(Renyi(q, base), probabilities(x, ε))
end

function genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_Dataset, est::ProbabilitiesEstimator; q, base)` is a deprecated function in
    Entropies.jl v2.0. Use instead: `entropy(Renyi(q, base), x, est)`,
    or `entropy(x, est)` if `q, base` have their default values.
    """
    return entropy(Renyi(q, base), x, est)
end

function symbolize(x::AbstractDataset{m, T}, est::PermutationProbabilityEstimator) where {m, T}
    Base.depwarn("`symbolize(x, est::$P)` is deprecated, use `symbolize(x, scheme::OrdinalPattern)` instead.", :symbolize)

    m >= 2 || error("Data must be at least 2-dimensional to symbolize. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    symbolize!(s, x, est)
    return s
end

function symbolize(x::AbstractVector{T}, est::P) where {T, P <: PermutationProbabilityEstimator}
    Base.depwarn("`symbolize(x, est::$P)` is deprecated, use `symbolize(x, scheme::OrdinalPattern)` instead.", :symbolize)

    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    symbolize!(s, x_emb, est)
    return s
end

function symbolize!(s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    Base.depwarn("`symbolize!(s, x, est::SymbolicPermutation)` is deprecated, use `symbolize!(s, x, scheme::OrdinalPattern)` instead.", :symbolize)

    @assert length(s) == length(x)
    #=
    Loop over embedding vectors `E[i]`, find the indices `p_i` that sort each `E[i]`,
    then get the corresponding integers `k_i` that generated the
    permutations `p_i`. Those integers are the symbols for the embedding vectors
    `E[i]`.
    =#
    sp = zeros(Int, m) # pre-allocate a single symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, m, lt = est.lt)

    return s
end

function symbolize!(s::AbstractVector{Int}, x::AbstractVector{T}, est::SymbolicPermutation) where T
    Base.depwarn("`symbolize!(s, x, est::SymbolicPermutation)` is deprecated, use `symbolize!(s, x, scheme::OrdinalPattern)` instead.", :symbolize)
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)
    symbolize!(s, x_emb, est)
end
