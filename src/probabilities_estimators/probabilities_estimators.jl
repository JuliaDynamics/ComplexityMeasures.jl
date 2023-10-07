function verify_counting_based(o, name = "BayesianRegularization")
    if !is_counting_based(o)
        s = "Outcome space $(o) isn't counting based."*
            "`$(name)` requires a counting-based outcome space."
        throw(ArgumentError(s))
    end
end

include("RelativeAmount.jl")
include("BayesianRegularization.jl")
include("Shrinkage.jl")
include("AddConstant.jl")

# Convenience, so that we can use the same syntax everywhere.
# --------------------------------------------------------------------------------
function counts(est::ProbabilitiesEstimator, outcomemodel::OutcomeSpace, x)
    return allcounts(outcomemodel, x)
end
function counts_and_outcomes(est::ProbabilitiesEstimator, outcomemodel::OutcomeSpace, x)
    return allcounts_and_outcomes(outcomemodel, x)
end
function allcounts(est::ProbabilitiesEstimator, outcomemodel::OutcomeSpace, x)
    return allcounts(outcomemodel, x)
end
function allcounts_and_outcomes(est::ProbabilitiesEstimator, outcomemodel::OutcomeSpace, x)
    return allcounts_and_outcomes(outcomemodel, x)
end
# These don't actually result in probability distributions.
#include("SampleCoverage.jl")
#include("GoodTuring.jl")
