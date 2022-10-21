export ValueHistogram, VisitationFrequency, AbstractBinning

"""
    AbstractBinning

The supertype of all binning schemes.
"""
abstract type AbstractBinning end

# We need binning to be defined first to add it as a field to a struct
include("rectangular_binning.jl")
include("histogram_estimation.jl")

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

    ValueHistogram(ϵ::Union{Real,Vector})

This is a convenience method that accepts same input as [`RectangularBinning`](@ref)
and initializes this binning directly.
"""
struct ValueHistogram{B<:AbstractBinning} <: ProbabilitiesEstimator
    binning::B
end
ValueHistogram(ϵ::Union{Real,Vector}) = ValueHistogram(RectangularBinning(ϵ))

"""
    VisitationFrequency

An alias for [`ValueHistogram`](@ref).
"""
const VisitationFrequency = ValueHistogram

# This method is only valid for rectangular binnings, as `fasthist`
# is only valid for rectangular binnings. For more binnings, it needs to be extended.
function probabilities(x::Array_or_Dataset, est::ValueHistogram{<:RectangularBinning})
    fasthist(x, est.binning)[1]
end

function probabilities_and_events(x, est::ValueHistogram)
    probs, bins, encoder = fasthist(x, est.binning)
    (; mini, edgelengths) = encoder
    unique!(bins) # `bins` is already sorted from `fasthist!`
    events = map(b -> b .* edgelengths .+ mini, bins)
    return probs, events
end
