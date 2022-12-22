import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation

"""
    SymbolicWeightedPermutation <: ProbabilitiesEstimator
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



[^Fadlallah2013]: Fadlallah, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical Review E 87.2
    (2013): 022911.
"""
struct SymbolicWeightedPermutation{F} <: PermutationProbabilitiesEstimator
    τ::Int
    m::Int
    lt::F
end
function SymbolicWeightedPermutation(; τ::Int = 1, m::Int = 3, lt::F = isless_rand) where {F <: Function}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    SymbolicWeightedPermutation{F}(τ, m, lt)
end

function permutation_weights(est::SymbolicWeightedPermutation, x::AbstractDataset)
    weights_from_variance.(x.data, est.m)
end
weights_from_variance(x, m::Int) = sum((x .- mean(x)) .^ 2)/m

probabilities(est::SymbolicWeightedPermutation, x) = encodings_and_probs(est, x)[2]
function probabilities_and_outcomes(est::SymbolicWeightedPermutation, x)
    encodings, probs = encodings_and_probs(est, x)
    return probs, observed_outcomes(est, encodings)
end


total_outcomes(est::SymbolicWeightedPermutation)::Int = factorial(est.m)
outcome_space(est::SymbolicWeightedPermutation) = permutations(1:est.m) |> collect
