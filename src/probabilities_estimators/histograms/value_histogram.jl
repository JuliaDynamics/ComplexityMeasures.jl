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
Available binnings are:
- [`RectangularBinning`](@ref)
- [`FixedRectangularBinning`](@ref)

The `ValueHistogram` estimator has a linearithmic time complexity
(`n log(n)` for `n = length(x)`) and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
For performance reasons,
the probabilities returned never contain 0s and are arbitrarily ordered.

    ValueHistogram(ϵ::Union{Real,Vector})

A convenience method that accepts same input as [`RectangularBinning`](@ref)
and initializes this binning directly.

## Outcomes

The outcome space for `ValueHistogram` is the unique bins constructed
from `b`. Each bin is identified by its left (lowest-value) corner.
The bins are in data units, not integer (cartesian indices units), and
are returned as `SVector`s.

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
# source code in truth is in the binnings file using the bin encoding

# This method is only valid for rectangular binnings, as `fasthist`
# is only valid for rectangular binnings. For more binnings, it needs to be extended.
function probabilities(x::Array_or_Dataset, est::ValueHistogram)
    # and the `fasthist` actually just makes an encoding,
    # this function is in `rectangular_binning.jl`
    Probabilities(fasthist(x, est.binning)[1])
end

function probabilities_and_outcomes(x::Array_or_Dataset, est::ValueHistogram)
    probs, bins, encoder = fasthist(x, est.binning)
    unique!(bins) # `bins` is already sorted from `fasthist!`
    # Here we transfor the cartesian coordinate based bins into data unit bins:
    outcomes = map(b -> decode_from_bin(b, encoder), bins)
    return Probabilities(probs), vec(outcomes)
end

outcome_space(x, est::ValueHistogram) = outcome_space(x, est.binning)
total_outcomes(x, est::ValueHistogram) = total_outcomes(x, est.binning)
