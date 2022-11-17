import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation

"""
    SymbolicWeightedPermutation(; τ = 1, m = 3, lt = Entropies.isless_rand)

A variant of [`SymbolicPermutation`](@ref) that also incorporates amplitude information,
based on the weighted permutation entropy (Fadlallah et al., 2013).

## Outcome space

Like for [`SymbolicPermutation`](@ref), the outcome space `Ω` for
`SymbolicWeightedPermutation` is the lexiographically ordered set of
length-`m` ordinal patterns (i.e. permutations) that can be formed by the integers
`1, 2, …, m`. There are `factorial(m)` such patterns.

## Description

Probabilities are computed as

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i}
\\left( \\mathbf{x}_k^{m, \\tau} \\right)
\\, w_k}{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi}
\\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,w_k} = \\dfrac{\\sum_{k=1}^N
\\mathbf{1}_{u:S(u) = s_i}
\\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, w_k}{\\sum_{k=1}^N w_k},
```

where weights are computed based on the variance of the state vectors as

```math
w_j = \\dfrac{1}{m}\\sum_{k=1}^m (x_{j+(k+1)\\tau} - \\mathbf{\\hat{x}}_j^{m, \\tau})^2,
```

and ``\\mathbf{x}_i`` is the aritmetic mean of state vector:

```math
\\mathbf{\\hat{x}}_j^{m, \\tau} = \\frac{1}{m} \\sum_{k=1}^m x_{j + (k+1)\\tau}.
```

The weighted permutation entropy is equivalent to regular permutation entropy when weights
are positive and identical (``w_j = \\beta \\,\\,\\, \\forall \\,\\,\\, j \\leq N`` and
``\\beta > 0)``.

See [`SymbolicPermutation`](@ref) for an estimator that only incorporates ordinal/sorting
information and disregards amplitudes, and [`SymbolicAmplitudeAwarePermutation`](@ref) for
another estimator that incorporates amplitude information.

!!! note "An implementation note"
    *Note: in equation 7, section III, of the original paper, the authors write*

    ```math
    w_j = \\dfrac{1}{m}\\sum_{k=1}^m (x_{j-(k-1)\\tau} - \\mathbf{\\hat{x}}_j^{m, \\tau})^2.
    ```
    *But given the formula they give for the arithmetic mean, this is **not** the variance
    of ``\\mathbf{x}_i``, because the indices are mixed: ``x_{j+(k-1)\\tau}`` in the weights
    formula, vs. ``x_{j+(k+1)\\tau}`` in the arithmetic mean formula. This seems to imply
    that amplitude information about previous delay vectors
    are mixed with mean amplitude information about current vectors. The authors also mix the
    terms "vector" and "neighboring vector" (but uses the same notation for both), making it
    hard to interpret whether the sign switch is a typo or intended. Here, we use the notation
    above, which actually computes the variance for ``\\mathbf{x}_i``*.



[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical Review E 87.2
    (2013): 022911.
"""
struct SymbolicWeightedPermutation{F} <: ProbabilitiesEstimator
    τ::Int
    m::Int
    lt::F
end
function SymbolicWeightedPermutation(; τ::Int = 1, m::Int = 3, lt::F = isless_rand) where {F <: Function}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    SymbolicWeightedPermutation{F}(τ, m, lt)
end

function weights_from_variance(x, m::Int)
    sum((x .- mean(x)) .^ 2)/m
end


function probabilities_and_outcomes(x::AbstractDataset{m, T},
        est::SymbolicWeightedPermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = outcomes(x, OrdinalPatternEncoding(m = m, lt = est.lt))  # motif length controlled by dimension of input data
    wts = weights_from_variance.(x.data, m)
    probs = symprobs(πs, wts, normalize = true)

    # The observed integer encodings are in the set `{0, 1, ..., factorial(m)}`, and each
    # integer corresponds to a unique permutation. Decoding an integer gives the original
    # permutation as a `SVector{m, Int}`.
    observed_encodings = sort(unique(πs))
    observed_outcomes = decode_motif.(observed_encodings, est.m)

   return Probabilities(probs), observed_outcomes
end

function probabilities_and_outcomes(x::AbstractVector{T},
        est::SymbolicWeightedPermutation) where {T<:Real}
    # We need to manually embed here instead of just calling the method above,
    # because the embedding vectors are needed to compute weights.
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = outcomes(x, OrdinalPatternEncoding(m = est.m, lt = est.lt)) # motif length controlled by estimator m
    wts = weights_from_variance.(emb.data, est.m)
    probs = symprobs(πs, wts, normalize = true)

    # The observed integer encodings are in the set `{0, 1, ..., factorial(m)}`, and each
    # integer corresponds to a unique permutation. Decoding an integer gives the original
    # permutation as a `SVector{m, Int}`.
    observed_encodings = sort(unique(πs))
    observed_outcomes = decode_motif.(observed_encodings, est.m)

    return Probabilities(probs), observed_outcomes
end

total_outcomes(est::SymbolicWeightedPermutation)::Int = factorial(est.m)
outcome_space(est::SymbolicWeightedPermutation) = permutations(1:est.m) |> collect
