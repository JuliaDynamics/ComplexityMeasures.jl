export ValueHistogram, VisitationFrequency, AbstractBinning

"""
    AbstractBinning

The supertype of all binning schemes.
"""
abstract type AbstractBinning end

# We need binning to be defined first to add it as a field to a struct
include("rectangular_binning.jl")
include("fasthist.jl")

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

## Outcomes

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

# For organizational outcomes we extend methods here. However, their
# source code in truth is in the binnings file using the encoding

# This method is only valid for rectangular binnings, as `fasthist`
# is only valid for rectangular binnings. For more binnings, it needs to be extended.
function probabilities(x::Array_or_Dataset, est::ValueHistogram)
    # and the `fasthist` actually just makes an encoding,
    # this function is in `rectangular_binning.jl`
    fasthist(x, est.binning)[1]
end

function probabilities_and_outcomes(x::Array_or_Dataset, est::ValueHistogram)
    probs, bins, encoder = fasthist(x, est.binning)
    unique!(bins) # `bins` is already sorted from `fasthist!`
    outcomes = map(b -> decode_from_bin(b, encoder), bins)
    return probs, outcomes
end

function outcome_space(x, est::ValueHistogram)
    encoder = RectangularBinEncoding(x, est.binning)
    return outcome_space(encoder)
end

function total_outcomes(x, est::ValueHistogram)
    encoder = RectangularBinEncoding(x, est.binning)
    return total_outcomes(encoder)
end
