export Jackknife

"""
    Jackknife <: DiscreteInfoEstimatorGeneric
    Jackknife(definition::InformationMeasure = Shannon())

The `Jackknife` estimator is used with [`information`](@ref) to compute any
discrete [`InformationMeasure`](@ref).

The `Jackknife` estimator uses the generic jackknife principle to reduce bias.
[Zahl1977](@citet) was the first to apply the jaccknife technique in the context of
[`Shannon`](@ref) entropy estimation. Here, we've generalized his estimator to work
with any [`InformationMeasure`](@ref).

## Description

As an example of the jackknife technique, here is the formula for a jackknife estimate
of [`Shannon`](@ref) entropy

```math
H_S^{J} = N H_S^{plugin} - \\dfrac{N-1}{N} \\sum_{i=1}^N {H_S^{plugin}}^{-\\{i\\}},
```

where ``N`` is the sample size, ``H_S^{plugin}`` is the plugin estimate of Shannon entropy,
and ``{H_S^{plugin}}^{-\\{i\\}}`` is the plugin estimate, but computed with the ``i``-th
sample left out.
"""
Base.@kwdef struct Jackknife{I <: InformationMeasure} <: DiscreteInfoEstimatorGeneric{I}
    definition::I = Shannon()
end

function information(hest::Jackknife, pest::ProbabilitiesEstimator,
        outcomemodel::OutcomeSpace, x)
    (; definition) = hest
    N = length(x)

    # The original estimate
    est_plugin = PlugIn(definition)
    i_plugin = information(est_plugin.definition, pest, outcomemodel, x)

    # The jackknifed estimates
    # TODO: this can be parallelized
    i_jackknifed = zeros(N)
    for i in eachindex(x)
        idxs = setdiff(1:N, i)
        xᵢ = @views x[idxs]
        i_jackknifed[i] = information(est_plugin.definition, pest, outcomemodel, xᵢ)
    end

    # The jackknifed estimate
    return N * i_plugin - (N - 1)/N * sum(i_jackknifed)
end
