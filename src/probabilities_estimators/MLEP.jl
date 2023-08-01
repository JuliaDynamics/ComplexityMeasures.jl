export MLEP

"""
    MLEP <: ProbabilitiesEstimator
    MLEP(outcome_space::OutcomeSpaceModel)

The `MLEP` estimator is used with [`probabilities`](@ref) to estimates
probabilities over the given [`OutcomeSpaceModel`](@ref) using empirical plug-in estimates
(also called "maximum likelihood estimation" probabilities; hence "MLEP").
"""
struct MLEP{O <: OutcomeSpaceModel} <: ProbabilitiesEstimator
    outcome_space::O
end

# Default to maximum likelihood estimation of probabilities if no other estimator is
# specified.
function probabilities_and_outcomes(o::OutcomeSpaceModel, x)
    return probabilities_and_outcomes(MLEP(o), x)
end

function probabilities_and_outcomes(est::MLEP, x)
    freqs, outcomes = frequencies_and_outcomes(outcome_space(est), x)
    return Probabilities(freqs), outcomes
end
