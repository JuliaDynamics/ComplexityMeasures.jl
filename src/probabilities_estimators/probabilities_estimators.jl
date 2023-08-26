function verify_counting_based(o, name = "BayesianRegularization")
    if !is_counting_based(o)
        s = "Outcome space $(o) isn't counting based."*
            "`$(name)` requires a counting-based outcome space."
        throw(ArgumentError(s))
    end
end

# The following methods may be overridden by individual `ProbabilitiesEstimator`s
# if doing so is more efficient.
# --------------------------------------------------------------------------------
function probabilities(est::ProbabilitiesEstimator, x)
    return first(probabilities_and_outcomes(est, x))
end

function allprobabilities(est::ProbabilitiesEstimator, x)
    return first(allprobabilities_and_outcomes(est, x))
end

include("RelativeAmount.jl")
include("BayesianRegularization.jl")
include("Shrinkage.jl")
include("AddConstant.jl")

# These don't actually result in probability distributions.
#include("SampleCoverage.jl")
#include("GoodTuring.jl")
