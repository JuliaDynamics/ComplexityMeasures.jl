using Combinatorics: permutations

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
    ``\\pi_{i}`` by comparing the elements in the vector. In this case, the values
    of `m, τ` are ignored.
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
struct SymbolicPermutation{F} <: PermutationProbabilitiesEstimator
    τ::Int
    m::Int
    lt::F
end
function SymbolicPermutation(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F <: Function}
    m >= 2 || throw(ArgumentError("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols."))
    SymbolicPermutation{F}(τ, m, lt)
end

function probabilities(est::SymbolicPermutation, x::Vector_or_Dataset)
    probabilities(encodings_from_permutations(est, x))
end

function probabilities!(πs::AbstractVector{Int}, est::SymbolicPermutation, x::Vector_or_Dataset)
    encodings_from_permutations!(πs, est, x)
    probabilities(πs)
end

function probabilities_and_outcomes(est::SymbolicPermutation, x::Vector_or_Dataset)
    πs = encodings_from_permutations(est, x)
    return probabilities(πs), observed_outcomes(est, πs)
end

function probabilities_and_outcomes!(πs::AbstractVector{Int}, est::SymbolicPermutation, x::Vector_or_Dataset)
    encodings_from_permutations!(πs, est, x)
    return probabilities(πs), observed_outcomes(est, πs)
end

function entropy!(
    s::AbstractVector{Int},
    e::Entropy,
    est::SymbolicPermutation,
    x::AbstractDataset{m, T};
    ) where {m, T}

    length(s) == length(x) || error("Pre-allocated symbol vector s need the same number of elements as x. Got length(πs)=$(length(πs)) and length(x)=$(L).")
    ps = probabilities!(s, est, x)

    entropy(e, ps)
end

function entropy!(
        s::AbstractVector{Int},
        e::Entropy,
        est::SymbolicPermutation,
        x::AbstractVector{T},
    ) where {T<:Real}

    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s` needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    ps = probabilities!(s, est, x)
    entropy(e, ps)
end

entropy!(s::AbstractVector{Int}, est::SymbolicPermutation, x) =
    entropy!(s, Shannon(base = 2), est, x)

total_outcomes(est::SymbolicPermutation)::Int = factorial(est.m)
outcome_space(est::SymbolicPermutation) = permutations(1:est.m) |> collect
