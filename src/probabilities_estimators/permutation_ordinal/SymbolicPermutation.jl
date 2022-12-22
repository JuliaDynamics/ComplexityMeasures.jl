export SymbolicPermutation

"""
    SymbolicPermutation <: ProbabilitiesEstimator
    SymbolicPermutation(; m = 3, τ = 1, lt::Function = Entropies.isless_rand)

A probabilities estimator based on ordinal permutation patterns.

When passed to [`probabilities`](@ref) the output depends on the input data type:

- **Univariate data**. If applied to a univariate timeseries (`Vector`), then the timeseries
    is first embedded using embedding delay `τ` and dimension `m`, resulting in embedding
    vectors ``\\{ \\bf{x}_i \\}_{i=1}^{N-(m-1)\\tau}``. Then, for each ``\\bf{x}_i``,
    we find its permutation pattern ``\\pi_{i}``. Probabilities are then
    estimated as the frequencies of the encoded permutation symbols
    by using [`CountOccurrences`](@ref). The resulting probabilities, when given to
    [`entropy`](@ref), compute the original permutation entropy[^BandtPompe2002].
- **Multivariate data**. If applied to a an `D`-dimensional `Dataset`,
    then no embedding is constructed. For each vector ``\\bf{x}_i``of the dataset,
    we directly map it to its permutation pattern
    Like above, probabilities are estimated as the frequencies of the permutation symbols.
    ``\\pi_{i}`` by comparing the elements in the vector. In this case, `m` is ignored,
    but `m` must still match the dimension of the dataset for optimization.
    The resulting probabilities can be used to compute multivariate permutation
    entropy[^He2016], although here we don't perform any further subdivision
    of the permutation patterns (as in Figure 3 of[^He2016]).

Internally, [`SymbolicPermutation`](@ref) uses the [`OrdinalPatternEncoding`](@ref)
to represent ordinal patterns as integers for efficient computations.

## Outcome space

The outcome space `Ω` for `SymbolicPermutation` is the set of length-`m` ordinal
patterns (i.e. permutations) that can be formed by the integers `1, 2, …, m`,
ordered lexicographically. There are `factorial(m)` such patterns.

For example, the outcome `[3, 1, 2]` corresponds to the ordinal pattern of having
first the largest value, then the lowest value, and then the value in between.

## In-place symbolization

`SymbolicPermutation` also implements the in-place [`entropy!`](@ref) and
[`probabilities!`](@ref). The length of the pre-allocated symbol vector must match the
length of the embedding: `N - (m-1)τ` for univariate timeseries, and `M` for length-`M`
`Dataset`s). For example

```julia
using DelayEmbeddings, Entropies
m, τ, N = 2, 1, 100
est = SymbolicPermutation(; m, τ)
x_ts = rand(N) # timeseries example
πs_ts = zeros(Int, N - (m - 1)*τ) # length must match length of delay embedding
p = probabilities!(πs_ts, est, x_ts)
h = entropy!(πs_ts, Renyi(), est, x_ts)
```

See [`SymbolicWeightedPermutation`](@ref) and [`SymbolicAmplitudeAwarePermutation`](@ref)
for estimators that not only consider ordinal (sorting) patterns, but also incorporate
information about within-state-vector amplitudes.

!!! note "Handling equal values in ordinal patterns"
    In Bandt & Pompe (2002), equal values are ordered after their order of appearance, but
    this can lead to erroneous temporal correlations, especially for data with
    low-amplitude resolution [^Zunino2017]. Here, by default, if two values are equal,
    then one of the is randomly assigned as "the largest", using
    `lt = Entropies.isless_rand`. To get the behaviour from Bandt and Pompe (2002), use
    `lt = Base.isless`).

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for timeseries." Physical review letters 88.17 (2002): 174102.
[^Zunino2017]: Zunino, L., Olivares, F., Scholkmann, F., & Rosso, O. A. (2017).
    Permutation entropy based timeseries analysis: Equalities in the input signal can
    lead to false conclusions. Physics Letters A, 381(22), 1883-1892.
[^He2016]:
    He, S., Sun, K., & Wang, H. (2016). Multivariate permutation entropy and its
    application for complexity analysis of chaotic systems. Physica A: Statistical
    Mechanics and its Applications, 461, 812-823.
"""
struct SymbolicPermutation{M,F} <: PermutationProbabilitiesEstimator
    encoding::OrdinalPatternEncoding{M,F}
    τ::Int
end
function SymbolicPermutation(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return SymbolicPermutation{m, F}(OrdinalPatternEncoding{m, F}(m, lt), τ)
end

# Probabilities etc. simply initialize the datasets and containers of the encodings
# and just map everythihng using `encode`.
function probabilities(est::SymbolicPermutation{m}, x::Vector{T}) where {m, T<:Real}
    dataset::Dataset{m,T} = genembed(x, m, est.τ)
    return probabilities(est, dataset)
end

function probabilities(est::SymbolicPermutation{m}, x::AbstractDataset{D}) where {m, D}
    m != D && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `Dataset` must match!"
    ))
    πs = zeros(Int, length(x))
    return probabilities!(πs, est, x)
end

function probabilities!(::Vector{Int}, ::SymbolicPermutation, ::AbstractVector)
    error("""
    In-place `probabilities!` for `SymbolicPermutation` can only be used by
    Dataset input, not timeseries. First embed the timeseries or use the
    normal version `probabilities`.
    """)
end

function _probabilities!(πs::Vector{Int}, est::SymbolicPermutation{m}, x::AbstractDataset{m}) where {m}
    # TODO: The following loop can probably be parallelized!
    @inbounds for (i, χ) in enumerate(x)
        πs[i] = encode(est.encoding, χ)
    end
    return Probabilities(fasthist!(πs))
end

function probabilities_and_outcomes(est::SymbolicPermutation{m}, x::Vector_or_Dataset) where {m}
    # A bit of code duplication here, because we actually need the processed
    # `πs` to invert it with `decode`. This can surely be optimized with some additional
    # function that both maps to integers with `decode` but also keeps track of
    # the permutation patter vectors. Anyways, I don't think `outcomes` is a function
    # that will be called often, so we can live with this as is.
    if x isa Vector
        dataset = genembed(x, m, est.τ)
    else
        dataset = x
    end
    m != dimension(dataset) && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `Dataset` must match!"
    ))
    πs = zeros(Int, length(dataset))
    @inbounds for (i, χ) in enumerate(dataset)
        πs[i] = encode(est.encoding, χ)
    end
    probs = Probabilities(fasthist!(πs))
    # Okay, now we compute the outcomes. (`πs` is already sorted in `fasthist!`)
    outcomes = decode.(Ref(est.encoding), unique!(πs))
    return probs, outcomes
end

# fallback
total_outcomes(est::PermutationProbabilitiesEstimator) = total_outcomes(est.encoding)
outcome_space(est::PermutationProbabilitiesEstimator) = outcome_space(est.encoding)
