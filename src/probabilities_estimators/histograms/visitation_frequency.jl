export VisitationFrequency, AbstractBinning

"""
    AbstractBinning

The supertype of all binning schemes.
"""
abstract type AbstractBinning end

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
struct VisitationFrequency{RB<:AbstractBinning} <: ProbabilitiesEstimator
    binning::RB
end

function probabilities(x::Array_or_Dataset, est::VisitationFrequency)
    probabilities(x, est.binning)
end

include("rectangular_binning.jl")
include("count_box_visits.jl")
include("histogram_estimation.jl")