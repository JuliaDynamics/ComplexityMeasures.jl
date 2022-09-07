export VisitationFrequency, probabilities
export entropy_visitfreq
export entropy_transferoperator

"""
    VisitationFrequency(r::RectangularBinning) <: BinningProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `r`.

## Example

```julia
# Construct boxes by dividing each coordinate axis into 5 equal-length chunks.
b = RectangularBinning(5)

# A probabilities estimator that, when applied a dataset, computes visitation frequencies
# over the boxes of the binning
est = VisitationFrequency(b)
```

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency{RB<:RectangularBinning} <: BinningProbabilitiesEstimator
    binning::RB
end

function probabilities(x::AbstractDataset, est::VisitationFrequency)
    _non0hist(x, est.binning)[1]
end

"""
    entropy_visitfreq(x, binning::RectangularBinning; base = MathConstants.e)

Compute the (Shannon) entropy of `x` by counting visitation frequencies over
the state-space coarse graining produced by the provided `binning` scheme.

Short-hand for `renyi_entropy(x, VisitationFrequency(binning); base = base, q = 1)`.

See also: [`VisitationFrequency`](@ref).
"""
function entropy_visitfreq(x, b::RectangularBinning)
    est = VisitationFrequency(binning)
    renyi_entropy(x, est; base = base, q = 1)
end
