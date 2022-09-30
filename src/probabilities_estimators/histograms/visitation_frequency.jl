export VisitationFrequency, AbstractBinning

"""
    AbstractBinning

The supertype of all binning schemes.
"""
abstract type AbstractBinning end

"""
    AbstractBinning

The supertype of all binning encoders, which, are used with `symbolize` to "encode"
a data point into its bin. The function `bin_encoder(x, b)` takes in an AbstractBinning
and returns the encoder.
"""
abstract type AbstractBinEncoder end


"""
    VisitationFrequency(b::AbstractBinning) <: ProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `b` and then computing the frequencies of points in the bins.

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

"""
    ValueHistogram
An alias for [`VisitationFrequency`](@ref).
"""
const ValueHistogram = VisitationFrequency

function probabilities(x::Array_or_Dataset, est::VisitationFrequency)
    probabilities(x, est.binning)
end
function probabilities(x::Array_or_Dataset)
    return Probabilities(fasthist(copy(x)))
end

include("rectangular_binning.jl")
include("histogram_estimation.jl")
include("count_box_visits.jl")