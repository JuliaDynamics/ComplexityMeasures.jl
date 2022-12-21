using Combinatorics: permutations

export SymbolicPermutation
"""
    SymbolicPermutation <: ProbabilitiesEstimator
    SymbolicPermutation(; m = 3, τ = 1, lt::Function = Entropies.isless_rand)

A probabilities estimator based on ordinal permutation patterns, originally used by
Bandt & Pompe (2002)[^BandtPompe2002] to compute permutation entropy.

If applied to a univariate time series, then the time series is first embedded using
embedding delay `τ` and dimension `m`, and then converted to a symbol time series using
[`outcomes`](@ref) with [`OrdinalPatternEncoding`](@ref), from which probabilities are
estimated. If applied to a `Dataset`, then `τ` and `m` are ignored, and probabilities are
computed directly from the state vectors.

## Outcome space

The outcome space `Ω` for `SymbolicPermutation` is the set of length-`m` ordinal
patterns (i.e. permutations) that can be formed by the integers `1, 2, …, m`,
ordered lexicographically. There are `factorial(m)` such patterns.

## In-place symbolization

`SymbolicPermutation` also implements the in-place [`entropy!`](@ref) and
[`probabilities!`](@ref). The length of the pre-allocated symbol vector must match the
length of the embedding: `N - (m-1)τ` for univariate time series, and `M` for length-`M`
`Dataset`s), i.e.

```julia
using DelayEmbeddings, Entropies
m, τ, N = 2, 1, 100
est = SymbolicPermutation(; m, τ)

# For a time series
x_ts = rand(N)
πs_ts = zeros(Int, N - (m - 1)*τ)
p = probabilities!(πs_ts, est, x_ts)
h = entropy!(πs_ts, Renyi(), est, x_ts)

# For a pre-discretized `Dataset`
x_symb = outcomes(x_ts, OrdinalPatternEncoding(m = 2, τ = 1))
x_d = genembed(x_symb, (0, -1, -2))
πs_d = zeros(Int, length(x_d))
p = probabilities!(πs_d, est, x_d)
h = entropy!(πs_d, Renyi(), est, x_d)
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
    complexity measure for time series." Physical review letters 88.17 (2002): 174102.
[^Zunino2017]: Zunino, L., Olivares, F., Scholkmann, F., & Rosso, O. A. (2017).
    Permutation entropy based time series analysis: Equalities in the input signal can
    lead to false conclusions. Physics Letters A, 381(22), 1883-1892.
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

function probabilities(est::SymbolicPermutation, x::AbstractDataset)
    πs = zeros(Int, length(x))
    probabilities(encodings_from_permutations!(πs, est, x))
end

function probabilities(est::SymbolicPermutation, x::AbstractVector)
    N = length(x) - (est.m - 1)*est.τ
    πs = zeros(Int, N)
    probabilities(encodings_from_permutations!(πs, est, x))
end

function probabilities!(πs::AbstractVector{Int}, est::SymbolicPermutation, x)
    encodings_from_permutations!(πs, est, x)
    probabilities(πs)
end

function probabilities_and_outcomes(est::SymbolicPermutation, x::AbstractDataset)
    πs = encodings_from_permutations(est, x)
    return probabilities(πs), observed_outcomes(est, πs)
end

function probabilities_and_outcomes(est::SymbolicPermutation, x::AbstractVector{<:Real})
    πs = encodings_from_permutations(est, x)
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
