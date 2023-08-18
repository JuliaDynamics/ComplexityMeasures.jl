export PlugIn

# Dummy estimator that doesn't actually change anything from the definitions
"""
    PlugIn(e::InformationMeasure) <: DiscreteInfoEstimator

The `PlugIn` estimator is also called the empirical/naive/"maximum likelihood" estimator,
and is used with [`information`](@ref) to any discrete [`InformationMeasure`](@ref).

It computes any quantity exactly as given by its formula. When computing an
information measure, which here is defined as a probabilities functional, it computes
the quantity directly from a probability mass function, which is derived from
maximum-likelihood ([`RelativeAmount`](@ref) estimates of the probabilities.

## Bias of plug-in estimates

The plugin-estimator of [`Shannon`](@ref) entropy underestimates the true entropy,
with a bias that grows with the number of distinct [`outcomes`](@ref) (Arora et al.,
2022)[^Arora2022]:

```math
bias(H_S^{plugin}) = -\\dfrac{K-1}{2N} + o(N^-1).
```

where `K` is the number of distinct outcomes, and `N` is the sample size. Many authors
have tried to remedy this by proposing alternative Shannon entropy estimators. For example,
the [`MillerMadow`](@ref) estimator is a simple correction to the plug-in estimator that
adds back the bias term above. Many other estimators exist; see
[`DiscreteInfoEstimator`](@ref)s for an overview.

[^Arora2022]:
    Arora, A., Meister, C., & Cotterell, R. (2022). Estimating the entropy of linguistic
    distributions. arXiv preprint arXiv:2204.01469.
"""
Base.@kwdef struct PlugIn{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
end

# Using the plugin-estimator is the same as plugging probabilities into the
# relevant definitions.
function information(hest::PlugIn, pest::ProbabilitiesEstimator, x)
    probs = probabilities(pest, x)
    return information(hest.definition, probs)
end
information(hest::PlugIn, probs::Probabilities) = information(hest.definition, probs)
information_normalized(e::PlugIn, est, x) = information_normalized(e.definition, est, x)
information_maximum(e::PlugIn, args...) = information_maximum(e.definition, args...)
