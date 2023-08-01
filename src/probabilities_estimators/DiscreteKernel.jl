export DiscreteKernel

"""
    DiscreteKernel{O <: OutcomeSpaceModel} <: ProbabilitiesEstimator
    DiscreteKernel(outcome_space::OutcomeSpaceModel)

"""
struct DiscreteKernel{O <: OutcomeSpaceModel} <: ProbabilitiesEstimator
    outcome_space::O
end
