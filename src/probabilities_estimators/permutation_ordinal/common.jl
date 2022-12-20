using StateSpaceSets: AbstractDataset
using StaticArrays: @MVector

# ----------------------------------------------------------------------------------------
# When given a dataset `X`, it is assumed that the dataset already represents an embedding.
# Therefore, each `xᵢ ∈ X` is directly encoded as an integer.
# ----------------------------------------------------------------------------------------
function encodings_from_permutations(est::PermutationProbabilitiesEstimator, x::AbstractDataset{m, T}) where {m, T}
    πs = zeros(Int, length(x))
    return encodings_from_permutations!(πs, est, x)
end

function encodings_from_permutations!(πs::AbstractVector{Int}, est::PermutationProbabilitiesEstimator,
        x::AbstractDataset{m, T}) where {m, T}
    if length(πs) != length(x)
        throw(ArgumentError("Need length(πs) == length(x), got `length(πs)=$(length(πs))` and `length(x)==$(length(x))`."))
    end

    encoding = OrdinalPatternEncoding(; m, lt = est.lt)
    perm = @MVector zeros(Int, 3)
    @inbounds for (i, xᵢ) in enumerate(x)
        sortperm!(perm, xᵢ, lt = est.lt)
        πs[i] = encode(encoding, perm)
    end
    return πs
end

# ----------------------------------------------------------------------------------------
# Timeseries are first embedded, then encoded as integers.
# ----------------------------------------------------------------------------------------
function encodings_from_permutations(est::PermutationProbabilitiesEstimator, x::AbstractVector{T}) where {T}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)
    N = length(x_emb)
    πs = zeros(Int, N)
    return encodings_from_permutations!(πs, est, x_emb)
end

function encodings_from_permutations!(πs::AbstractVector{Int}, est::PermutationProbabilitiesEstimator,
        x::AbstractVector{T}) where T
    N = length(x) - (est.m - 1)*est.τ
    length(πs) == N || throw(ArgumentError("Pre-allocated symbol vector `πs` needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(πs)=$(length(πs)) and length(x)=$(L)."))
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)
    encodings_from_permutations!(πs, est, x_emb)
end

# ----------------------------------------------------------------------------------------
# `SymbolicPermutation`, `SymbolicWeightedPermutation` and `SymbolicPermutation` are
# essentially identical, but weights are computed differently (and no weights are used for
# `SymbolicPermutation`). Therefore, we define a common `permutation_weights` function,
# so we don't need to duplicate code.
# ----------------------------------------------------------------------------------------
function permutation_weights end

function encodings_and_probs(
        est::PermutationProbabilitiesEstimator,
        x::AbstractDataset
    )
    πs = encodings_from_permutations(est, x)
    wts = permutation_weights(est, x)
    return πs, Probabilities(symprobs(πs, wts, normalize = true))
end

function encodings_and_probs(
        est::PermutationProbabilitiesEstimator,
        x::AbstractVector{<:Real}
    )
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = encodings_from_permutations(est, x)
    wts = permutation_weights(est, emb)
    return πs, Probabilities(symprobs(πs, wts, normalize = true))
end

function observed_outcomes(est::PermutationProbabilitiesEstimator, encodings)
    encoding = OrdinalPatternEncoding(m = est.m, lt = est.lt)
    observed_encodings = sort(unique(encodings))
    return decode.(Ref(encoding), observed_encodings) # int → permutation
end
