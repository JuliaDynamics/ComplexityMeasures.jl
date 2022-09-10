export VisitationFrequency, probabilities

abstract type BinningProbabilitiesEstimator <: ProbabilitiesEstimator end

"""
    VisitationFrequency(r::RectangularBinning) <: ProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `r` and then computing the frequencies of points in the bins.

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency{RB<:RectangularBinning} <: BinningProbabilitiesEstimator
    binning::RB
end

function probabilities(x::AbstractDataset, est::VisitationFrequency)
    _non0hist(x, est.binning)[1]
end
