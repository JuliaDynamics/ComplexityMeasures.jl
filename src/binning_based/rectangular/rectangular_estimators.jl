abstract type BinningProbabilitiesEstimator <: ProbabilitiesEstimator end

include("binning_schemes.jl")

include("utils.jl")
include("count_box_visits.jl")
include("histogram_estimation.jl")

# Estimators
include("VisitationFrequency.jl")