export VisitationFrequency, probabilities

abstract type BinningProbabilitiesEstimator <: ProbabilitiesEstimator end

"""
    VisitationFrequency(r::RectangularBinning) <: ProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `r` and then computing the frequencies of points in the bins.

This method has a linearithmic time complexity (`n log(n)` for `n = length(x)`)
and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
To obtain the bin information along with the probabilities, use [`binhist`](@ref).

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency{RB<:RectangularBinning} <: BinningProbabilitiesEstimator
    binning::RB
end

function probabilities(x::AbstractDataset, est::VisitationFrequency)
    _non0hist(x, est.binning)[1]
end
