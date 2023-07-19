export ProbabilitiesFunctional
export ProbabilitiesFunctionalEstimator

"""
    ProbabilitiesFunctional

The supertype of all probabilities functional *definitions*. To keep naming conventions
simply, we also use this as the supertype of *probability density functionals*.

## Abstract subtypes

- [`InformationMeasureDefinition`](@ref). All probabilities functionals that aim to quantify
    information in some way, typically some form of entropy, but also extropies and
    other functionals of probabilities/densities.
"""
abstract type ProbabilitiesFunctional end

"""
    ProbabilitiesFunctionalEstimator

The supertype of all probabilities functionals *estimators*. To keep naming conventions
simply, we also use this as the supertype of *probability density functional* estimators.
"""
abstract type ProbabilitiesFunctionalEstimator end
