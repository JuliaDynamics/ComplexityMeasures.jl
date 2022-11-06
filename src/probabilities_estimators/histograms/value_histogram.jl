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
the frequencies of points in the bins. An alias to this is `VisitationFrequency`.

    ValueHistogram(ϵ::Union{Real,Vector})

A convenience method that accepts same input as [`RectangularBinning`](@ref)
and initializes this binning directly.

The `ValueHistogram` estimator has a linearithmic time complexity
(`n log(n)` for `n = length(x)`) and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

# Outcomes

The outcomes `Ω` for `ValueHistogram` is the set of unique bins constructed
from `b`. Each bin is identified by its left (lowest-value) corner.
Use [`probabilities_and_outcomes`](@ref) to obtain bins together
with the probabilities.

See also: [`RectangularBinning`](@ref).
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
function probabilities(x::Array_or_Dataset, est::ValueHistogram)
    fasthist(x, est.binning)[1]
end

function probabilities_and_outcomes(x::Array_or_Dataset, est::ValueHistogram)
    probs, bins, encoder = fasthist(x, est.binning)
    (; mini, edgelengths) = encoder
    unique!(bins) # `bins` is already sorted from `fasthist!`
    events = map(b -> b .* edgelengths .+ mini, bins)
    return probs, events
end

function all_possible_outcomes(x::Array_or_Dataset, est::ValueHistogram)
    encoder = RectangularBinEncoding(x, est.binning)
    return all_possible_outcomes(encoder)
end
