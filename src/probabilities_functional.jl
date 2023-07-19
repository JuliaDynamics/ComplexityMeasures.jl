export ProbabilitiesFunctional
export ProbabilitiesFunctionalEstimator

"""
    ProbabilitiesFunctional

The supertype of all probabilities functional *definitions*. To keep naming conventions
simply, we use this type as the supertype of both *probability mass functionals*, as well
as *probability density functionals*.

## Abstract subtypes

- [`InformationMeasureDefinition`](@ref). All probabilities functionals that aim to quantify
    information in some way, typically some form of entropy, but also extropies and
    other functionals of probabilities/densities.
"""
abstract type ProbabilitiesFunctional end

"""
    ProbabilitiesFunctionalEstimator

The supertype of all *estimators* that estimate some [`ProbabilitiesFunctional`](@ref),
which can either refer to a probability mass functional, or a probability density
functional.
"""
abstract type ProbabilitiesFunctionalEstimator end
