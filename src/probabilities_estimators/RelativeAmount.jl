export RelativeAmount

"""
    RelativeAmount <: ProbabilitiesEstimator
    RelativeAmount(o::OutcomeSpace)

The `RelativeAmount` estimator is used with [`probabilities`](@ref) and related functions to estimate
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

This estimation is known as _Maximum Likelihood Estimation_.
However, `RelativeAmount` also serves as the fall-back probabilities estimator for [`OutcomeSpace`](@ref)s
that are not count-based and only yield "pseudo-counts", for example [`WaveletOverlap`](@ref) or
[`PowerSpectrum`](@ref). These outcome spaces do not yield counts, but pre-normalized
numbers that can be treated as "relative frequencies" or "relative power".
Hence, this estimator is called `RelativeAmount`.

## Examples

```julia
using ComplexityMeasures
x = cumsum(randn(100))
ps = probabilities(OrdinalPatterns(m = 3), x) # RelativeAmount is the default estimator
ps_mle = probabilities(RelativeAmount(OrdinalPatterns(m = 3)), x) # equivalent
ps == ps_mle # true
```

See also: [`Bayes`](@ref), [`Shrinkage`](@ref).
"""
struct RelativeAmount{O <: OutcomeSpace} <: ProbabilitiesEstimator
    outcomemodel::O
end

probabilities(est::RelativeAmount, x) = probabilities(est.outcomemodel, x)
allprobabilities(est::RelativeAmount, x) = allprobabilities(est.outcomemodel, x)
counts_and_outcomes(est::RelativeAmount, x) = counts_and_outcomes(est.outcomemodel, x)
counts(est::RelativeAmount, x) = counts(est.outcomemodel, x)

function probabilities_and_outcomes(est::RelativeAmount, x)
    return probabilities_and_outcomes(est.outcomemodel, x)
end

function allprobabilities_and_outcomes(est::RelativeAmount, x)
    return allprobabilities_and_outcomes(est.outcomemodel, x)
end
