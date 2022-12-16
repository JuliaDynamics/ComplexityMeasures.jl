export SymbolicAmplitudeAwarePermutation

"""
    SymbolicAmplitudeAwarePermutation(; τ = 1, m = 3, A = 0.5, lt = Entropies.isless_rand)

A variant of [`SymbolicPermutation`](@ref) that also incorporates amplitude information,
based on the amplitude-aware permutation entropy (Azami & Escudero, 2016).

## Outcome space

Like for [`SymbolicPermutation`](@ref), the outcome space `Ω` for
`SymbolicAmplitudeAwarePermutation` is the lexiographically ordered set of
length-`m` ordinal patterns (i.e. permutations) that can be formed by the integers
`1, 2, …, m`. There are `factorial(m)` such patterns.

## Description

Probabilities are computed as

```math
p(\\pi_i^{m, \\tau}) =
\\dfrac{\\sum_{k=1}^N
\\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N
\\mathbf{1}_{u:S(u) \\in \\Pi} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,a_k} =
\\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i}
\\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N a_k}.
```

The weights encoding amplitude information about state vector
``\\mathbf{x}_i = (x_1^i, x_2^i, \\ldots, x_m^i)`` are

```math
a_i = \\dfrac{A}{m} \\sum_{k=1}^m |x_k^i | + \\dfrac{1-A}{d-1}
\\sum_{k=2}^d |x_{k}^i - x_{k-1}^i|,
```

with ``0 \\leq A \\leq 1``. When ``A=0`` , only internal differences between the
elements of
``\\mathbf{x}_i`` are weighted. Only mean amplitude of the state vector
elements are weighted when ``A=1``. With, ``0<A<1``, a combined weighting is used.

See [`SymbolicPermutation`](@ref) for an estimator that only incorporates ordinal/sorting
information and disregards amplitudes, and [`SymbolicWeightedPermutation`](@ref) for
another estimator that incorporates amplitude information.

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy:
    Illustration in spike detection and signal segmentation. Computer methods and programs
    in biomedicine, 128, 40-51.
"""
struct SymbolicAmplitudeAwarePermutation{F} <: ProbabilitiesEstimator
    τ::Int
    m::Int
    A::Float64
    lt::F
end
function SymbolicAmplitudeAwarePermutation(; τ::Int = 1, m::Int = 2, A::Real = 0.5,
        lt::F = isless_rand) where {F <: Function}
    2 ≤ m || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    0 ≤ A ≤ 1 || error("Weighting factor A must be on interval [0, 1]. Got A=$A.")
    SymbolicAmplitudeAwarePermutation{F}(τ, m, A, lt)
end

"""
    AAPE(x, A::Real = 0.5, m::Int = length(a))

Encode relative amplitude information of the elements of `a`.
- `A = 1` emphasizes only average values.
- `A = 0` emphasizes changes in amplitude values.
- `A = 0.5` equally emphasizes average values and changes in the amplitude values.
"""
function AAPE(x; A::Real = 0.5, m::Int = length(x))
    (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

function probabilities_and_outcomes(est::SymbolicAmplitudeAwarePermutation,
        x::AbstractDataset{m, T}) where {m, T}
    πs = outcomes(x, OrdinalPatternEncoding(m = m, lt = est.lt))
    wts = AAPE.(x.data, A = est.A, m = est.m)
    probs = symprobs(πs, wts, normalize = true)

    # The observed integer encodings are in the set `{0, 1, ..., factorial(m)}`, and each
    # integer corresponds to a unique permutation. Decoding an integer gives the original
    # permutation as a `SVector{m, Int}`.
    observed_encodings = sort(unique(πs))
    observed_outcomes = decode_motif.(observed_encodings, est.m)

    return Probabilities(probs), observed_outcomes
end

function probabilities_and_outcomes(
        est::SymbolicAmplitudeAwarePermutation,
        x::AbstractVector{T}) where {T<:Real}
    # We need to manually embed here instead of just calling the method above,
    # because the embedding vectors are needed to compute weights.
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = outcomes(emb, OrdinalPatternEncoding(m = est.m, lt = est.lt))
    wts = AAPE.(emb.data, A = est.A, m = est.m)
    probs = symprobs(πs, wts, normalize = true)

    # The observed integer encodings are in the set `{0, 1, ..., factorial(m)}`, and each
    # integer corresponds to a unique permutation. Decoding an integer gives the original
    # permutation as a `SVector{m, Int}`.
    observed_encodings = sort(unique(πs))
    observed_outcomes = decode_motif.(observed_encodings, est.m)

    return Probabilities(probs), observed_outcomes
end

total_outcomes(est::SymbolicAmplitudeAwarePermutation)::Int = factorial(est.m)
outcome_space(est::SymbolicAmplitudeAwarePermutation) = permutations(1:est.m) |> collect
