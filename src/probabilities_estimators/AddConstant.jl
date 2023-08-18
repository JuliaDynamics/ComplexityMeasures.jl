export AddConstant

"""
    AddConstant <: ProbabilitiesEstimator
    AddConstant(o::OutcomeSpace, c = 1.0)

A generic add-constant probabilities estimator for counting-based [`OutcomeSpace`](@ref)s,
where several literature estimators can be obtained tuning `c`. Currently ``c``
can only be a scalar.

- `c = 1.0` is the Laplace estimator, or the "add-one" estimator.

## Description

Probabilities for the ``k``-th outcome ``\\omega_{k}`` are estimated as

```math
p(\\omega_k) = \\dfrac{(n_k + c)}{n + mc},
```

where ``m`` is the cardinality of the outcome space, and ``n`` is the number of
(encoded) input data points, and ``n_k`` is the number of times the outcome ``\\omega_{k}``
is observed in the (encoded) input data points.

If the `AddConstant` estimator used with
[`probabilities`](@ref)/[`probabilities_and_outcomes`](@ref),
then ``m`` is set to the number of *observed* outcomes. If used with
[`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref), then ``m`` is set to
the number of *possible* outcomes.

!!! note "Unobserved outcomes are assigned nonzero probability!"
    Looking at the formula above, if ``n_k = 0``, then unobserved outcomes are assigned
    a non-zero probability of ``\\dfrac{c}{n + mc}``. This means that if the estimator
    is used with [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref), then
    all outcomes, even those that are not observed, are assigned non-zero probabilities.
    This might affect your results if using e.g. [`missing_outcomes`](@ref).
"""
struct AddConstant{O <: OutcomeSpace, A} <: ProbabilitiesEstimator
    outcomemodel::O
    c::A

    function AddConstant(o::O, c::A) where {O <: OutcomeSpace, A}
        verify_counting_based(o)
        new{O, A}(o, c)
    end
end
AddConstant(outcomemodel::OutcomeSpace; c = 1.0) = AddConstant(outcomemodel, c)

function probabilities_and_outcomes(est::AddConstant, x)
    cts, Ω_observed = counts_and_outcomes(est.outcomemodel, x) # Ω_observed is now Ω.
    return probs_and_outs_from_histogram(est, cts, Ω_observed, x)
end

# Assigns non-zero probability to unobserved outcomes.
function allprobabilities_and_outcomes(est::AddConstant, x)
    cts, Ω_observed = allcounts_and_outcomes(est.outcomemodel, x) # Ω_observed is now Ω.
    return probs_and_outs_from_histogram(est, cts, Ω_observed, x)
end

function probs_and_outs_from_histogram(est::AddConstant, cts_observed, Ω_observed, x)
    (; outcomemodel, c) = est
    m = length(Ω_observed)
    n = encoded_space_cardinality(outcomemodel, x) # Normalize based on *encoded* data.
    probs = zeros(m)
    for (k, nₖ) in enumerate(cts_observed)
        probs[k] = (nₖ + c) / (n + (c * m))
    end
    @assert sum(probs) ≈ 1

    return Probabilities(probs), Ω_observed
end
