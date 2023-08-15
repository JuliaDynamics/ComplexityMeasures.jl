export MLE

"""
    MLE <: ProbabilitiesEstimator
    MLE(o::OutcomeSpace)

The `MLE` estimator is used with [`probabilities`](@ref) and related functions to estimate
probabilities over the given [`OutcomeSpace`](@ref) using maximum likelihood estimation
(MLE), also called plug-in estimation. See [`ProbabilitiesEstimator`](@ref) for usage.

## Description

Consider a length-`m` outcome space ``\\Omega`` and random sample of length `N`.
The maximum likelihood estimate of the probability of the `k`-th outcome ``\\omega_k`` is

```math
p(\\omega_k) = \\dfrac{n_k}{N},
```
where ``n_k`` is the number of times the `k`-th outcome was observed in the (encoded)
sample.

`MLE` also serves as the fall-back probabilities estimator for [`OutcomeSpace`](@ref)s
that only yield "pseudo-counts", for example [`WaveletOverlap`](@ref) or
[`PowerSpectrum`](@ref). These outcome spaces do not yield counts, but pre-normalized
numbers that can be treated as "relative frequencies".

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps = probabilities(SymbolicPermutation(m = 3), x) # MLE is the default estimator
ps_mle = probabilities(MLE(SymbolicPermutation(m = 3)), x) # equivalent
ps == ps_mle # true
```

See also: [`Bayes`](@ref), [`Shrinkage`](@ref).
"""
struct MLE{O <: OutcomeSpace} <: ProbabilitiesEstimator
    outcomemodel::O
end

probabilities(est::MLE, x) = probabilities(est.outcomemodel, x)
allprobabilities(est::MLE, x) = allprobabilities(est.outcomemodel, x)
counts_and_outcomes(est::MLE, x) = counts_and_outcomes(est.outcomemodel, x)
counts(est::MLE, x) = counts(est.outcomemodel, x)

function probabilities_and_outcomes(est::MLE, x)
    return probabilities_and_outcomes(est.outcomemodel, x)
end

function allprobabilities_and_outcomes(est::MLE, x)
    return allprobabilities_and_outcomes(est.outcomemodel, x)
end
