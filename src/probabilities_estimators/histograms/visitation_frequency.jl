export ValueHistogram, VisitationFrequency, AbstractBinning

"""
    AbstractBinning

The supertype of all binning schemes.
"""
abstract type AbstractBinning end

"""
    ValueHistogram(b::AbstractBinning) <: ProbabilitiesEstimator

A probability estimator based on binning the values of the data as dictated by
the binning scheme `b` and formally computing their histogram, i.e.,
the frequencies of points in the bins. Alias to this is `VisitationFrequency`.

This method has a linearithmic time complexity (`n log(n)` for `n = length(x)`)
and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

To obtain the bin information along with the probabilities,
use [`probabilities_and_events`](@ref). The events correspond to the bin corners.

See also: [`RectangularBinning`](@ref).
"""
struct ValueHistogram{RB<:AbstractBinning} <: ProbabilitiesEstimator
    binning::RB
end

"""
    VisitationFrequency

An alias for [`ValueHistogram`](@ref).
"""
const VisitationFrequency = ValueHistogram

function probabilities(x::Array_or_Dataset, est::ValueHistogram)
    probabilities(x, est.binning)
end

function probabilities_and_events(x, est::ValueHistogram)
    return probabilities_and_events(x, est.binning)
end

include("rectangular_binning.jl")
include("histogram_estimation.jl")
