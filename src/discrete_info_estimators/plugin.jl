export PlugIn

# Dummy estimator that doesn't actually change anything from the definitions
"""
    PlugIn(e::InformationMeasure) <: DiscreteInfoEstimator

The `PlugIn` estimator is also called the empirical/naive/"maximum likelihood" estimator.

This estimator calculates a quantity exactly as given by its formula, using plug-in
estimates (i.e. observed frequencies). For information measures, for example, it
estimates the [`InformationMeasure`](@ref) directly from a probability mass
function (which is derived from plug-in estimates of the probabilities).
"""
Base.@kwdef struct PlugIn{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    definition::I = Shannon()
end

# Using the plugin-estimator is the same as plugging probabilities into the
# relevant definitions.
function information(hest::PlugIn, pest::ProbabilitiesEstimator, x)
    return information(hest.definition, pest, x)
end
information(hest::PlugIn, probs::Probabilities) = information(hest.definition, probs)
information_normalized(e::PlugIn, est, x) = information_normalized(e.definition, est, x)
information_maximum(e::PlugIn, args...) = information_maximum(e.definition, args...)
