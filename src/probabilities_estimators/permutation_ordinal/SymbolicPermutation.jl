export SymbolicPermutation

"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilityEstimator <: ProbabilitiesEstimator end

"""
    SymbolicPermutation(; m = 3, τ = 1, lt::Function = Entropies.isless_rand)

A probabilities estimator based on ordinal permutation patterns, originally used by
Bandt & Pompe (2002)[^BandtPompe2002] to compute permutation entropy.

If applied to a univariate time series, then the time series is first embedded using
embedding delay `τ` and dimension `m`, and then converted to a symbol time series using
[`outcomes`](@ref) with [`OrdinalMapping`](@ref), from which probabilities are
estimated. If applied to a `Dataset`, then `τ` and `m` are ignored, and probabilities are
computed directly from the state vectors.

## Outcomes

The outcome space `Ω` for `SymbolicPermutation` is the set `{1, 2, …, factorial(m)}`,
where each integer correspond to a unique ordinal pattern, but
[`probabilities_and_outcomes`](@ref) is not yet implemented for this estimator.

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
s_ts = zeros(Int, N - (m - 1)*τ)
p = probabilities!(s_ts, x_ts, est)
h = entropy!(s_ts, Renyi(),  x_ts, est)

# For a pre-discretized `Dataset`
x_symb = outcomes(x_ts, OrdinalMapping(m = 2, τ = 1))
x_d = genembed(x_symb, (0, -1, -2))
s_d = zeros(Int, length(x_d))
p = probabilities!(s_d, x_d, est)
h = entropy!(s_d, Renyi(), x_d, est)
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
struct SymbolicPermutation{F} <: PermutationProbabilityEstimator
    τ::Int
    m::Int
    lt::F
end
function SymbolicPermutation(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F <: Function}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    SymbolicPermutation{F}(τ, m, lt)
end

function probabilities!(s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    length(s) == length(x) || throw(ArgumentError("Need length(s) == length(x), got `length(s)=$(length(s))` and `length(x)==$(length(x))`."))
    m >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series embed it using `genembed` first.")

    @inbounds for i in eachindex(x)
        s[i] = encode_motif(x[i], m)
    end
    probabilities(s)
end

function probabilities!(s::AbstractVector{Int}, x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s`needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    probabilities!(s, x_emb, est)
end

function probabilities(x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    s = zeros(Int, length(x))
    probabilities!(s, x, est)
end

function probabilities(x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    probabilities!(s, x_emb, est)
end

function entropy!(e::Entropy,
    s::AbstractVector{Int},
    x::AbstractDataset{m, T},
    est::SymbolicPermutation;
    ) where {m, T}

    length(s) == length(x) || error("Pre-allocated symbol vector s need the same number of elements as x. Got length(s)=$(length(s)) and length(x)=$(L).")
    ps = probabilities!(s, x, est)

    entropy(e, ps)
end

function entropy!(e::Entropy,
        s::AbstractVector{Int},
        x::AbstractVector{T},
        est::SymbolicPermutation
    ) where {T<:Real}

    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s` needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    ps = probabilities!(s, x, est)
    entropy(e, ps)
end

total_outcomes(est::SymbolicPermutation)::Int = factorial(est.m)
