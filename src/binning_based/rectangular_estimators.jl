abstract type BinningProbabilitiesEstimator <: ProbabilitiesEstimator end

include("GroupSlices.jl")
include("binning_schemes.jl")
include("visitation_frequency/VisitationFrequency.jl")
include("transferoperator/transfer_operator.jl")