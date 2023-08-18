export Bayes

"""
    Bayes <: ProbabilitiesEstimator
    Bayes(outcome_space::OutcomeSpace, a = 1.0)

The `BayesProbs` estimator is used with [`probabilities`](@ref) and related functions to
estimate probabilities over the given `m`-element counting-based [`OutcomeSpace`](@ref)
using Bayesian regularization of cell counts (Hausser & Strimmer, 2009)[^Hausser2009].
See [`ProbabilitiesEstimator`](@ref) for usage.

## Outcome space requirements

This estimator only works with counting-compatible outcome spaces.

## Description

The `Bayes` estimator estimates the probability of the ``k``-th outcome ``\\omega_{k}``
is

```math
\\omega_{k}^{\\text{Bayes}} = \\dfrac{n_k + a_k}{n + A},
```

where ``n`` is the number of samples in the input data, ``n_k`` is the observed counts
for the outcome ``\\omega_{k}``, and ``A = \\sum_{i=1}^k a_k``.

## Picking `a`

There are many common choices of priors, some of which are listed in
Hausser & Strimmer (2009)[^Hausser2009]. They include

- `a == 0`, which is equivalent to the [`RelativeAmount`](@ref) estimator.
- `a == 0.5` (Jeffrey's prior)
- `a == 1` (Bayes-Laplace uniform prior)

`a` can also be chosen as a vector of real numbers. Then, if used with
[`allprobabilities`](@ref), it is required that `length(a) == total_outcomes(o, x)`,
where `x` is the input data and `o` is the [`OutcomeSpace`](@ref).
If used with [`probabilities`](@ref), then `length(a)` must match the number of
*observed* outcomes (you can check this using [`probabilities_and_outcomes`](@ref)).
The choice of `a` can severely impact the estimation errors of the probabilities,
and the errors depend both on the choice of `a` and on the sampling scenario[^Hausser2009].

## Assumptions

The `Bayes` estimator assumes a fixed and known `m`. Thus, using it with
[`probabilities`](@ref) and [`allprobabilities`](@ref) will yield different results,
depending on whether all outcomes are observed in the input data or not.
For [`probabilities`](@ref), `m` is the number of *observed* outcomes.
For [`allprobabilities`](@ref), `m = total_outcomes(o, x)`, where `o` is the
[`OutcomeSpace`](@ref) and `x` is the input data.

!!! note
    If used with [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref), then
    outcomes which have not been observed may be assigned non-zero probabilities.
    This might affect your results if using e.g. [`missing_outcomes`](@ref).

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps_bayes = probabilities(Bayes(SymbolicPermutation(m = 3), a = 0.5), x)
```

See also: [`RelativeAmount`](@ref), [`Shrinkage`](@ref).

[^Hausser2009]:
    Hausser, J., & Strimmer, K. (2009). Entropy inference and the James-Stein estimator,
    with application to nonlinear gene association networks. Journal of Machine Learning
    Research, 10(7).
"""
struct Bayes{O <: OutcomeSpace, A} <: ProbabilitiesEstimator
    outcomemodel::O
    a::A
    function Bayes(o::O, c::A) where {O <: OutcomeSpace, A}
        verify_counting_based(o, "Bayes")
        new{O, A}(o, c)
    end
end
Bayes(outcomemodel::OutcomeSpace; a = 1.0) = Bayes(outcomemodel, a)

# We need to implement `probabilities_and_outcomes` and `allprobabilities` separately,
# because the number of elements in the outcome space determines the factor `A`, since
# A = sum(aₖ). Explicitly modelling the entire outcome space, instead of considering
# only the observed outcomes, will therefore affect the estimated probabilities.
function probabilities_and_outcomes(est::Bayes, x)
    (; outcomemodel, a) = est

    observed_counts, observed_outcomes = counts_and_outcomes(outcomemodel, x)
    M = length(observed_outcomes)

    if a isa Vector{<:Real}
        length(a) == M || throw(DimensionMismatch("length(a) must equal the number of elements in the observed outcome space (got $M outcomes, but length(a)=$(length(a)))."))
    end

    # Estimate posterior distribution.
    probs = zeros(M)
    A = sum(a)

    # Normalization factor is based on the *encoded* data.
    n = encoded_space_cardinality(outcomemodel, x)

    # Estimate probability for each observed outcome.
    for i = 1:M
        yᵢ = observed_counts[i]
        aₖ = get_aₖ_bayes(a, i)
        probs[i] = θ̂bayes(yᵢ, aₖ, n, A)
    end

    return Probabilities(probs), observed_outcomes
end

function allprobabilities_and_outcomes(est::Bayes, x)
    (; outcomemodel, a) = est

    # Normalization factor is based on the *encoded* data.
    n = encoded_space_cardinality(outcomemodel, x)

    a = est.a
    Ω = outcome_space(outcomemodel, x)
    M = length(Ω)
    if a isa Vector{<:Real}
        length(a) == M || throw(DimensionMismatch("length(a) must equal the number of " *
        "elements in the outcome space (got $M outcomes, but length(a)=$(length(a)))."))
    end
    observed_counts, observed_outcomes = counts_and_outcomes(est.outcomemodel, x)

    # Estimate posterior distribution.
    probs = zeros(M)
    A = sum(a)
    for i in eachindex(observed_outcomes)
        Ωᵢ = observed_outcomes[i]
        yᵢ = observed_counts[i]
        idx = findfirst(x -> x == Ωᵢ, Ω)
        aₖ = get_aₖ_bayes(a, i)
        probs[idx] = θ̂bayes(yᵢ, aₖ, n, A)
    end

    return Probabilities(probs), Ω
end

get_aₖ_bayes(a, i) = a isa Real ? a : a[i]
θ̂bayes(yₖ, aₖ, n, A) = (yₖ + aₖ) / (n + A)
