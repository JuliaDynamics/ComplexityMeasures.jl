export VisitationFrequency, probabilities, genentropy
import DelayEmbeddings: Dataset, AbstractDataset

"""
    VisitationFrequency(r::RectangularBinning)

A probability estimator based on binning data into rectangular boxes dictated by 
the binning scheme `r`.

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency <: BinningProbabilitiesEstimator
    binning::RectangularBinning
    
    function VisitationFrequency(r::RectangularBinning)
        new(r)
    end
end

"""
# Visitation frequency, binning based probabilities

    probabilities(x::Dataset, est::VisitationFrequency) → Vector{Real}

Superimpose a rectangular grid (bins/boxes) dictated by `est` over the data `x` and return 
the sum-normalized histogram (i.e. frequency at which the points of `x` visits the bins/boxes 
in the grid) in an unordered 1D form, discarding all non-visited bins and bin edge information.

# Performances Notes
    
This method has a linearithmic time complexity `(n log(n)` for `n = length(data)`) and a 
linear space complexity `l` for `l = dimension(data)`). This allows computation of 
histograms of high-dimensional datasets and with small box sizes ε without memory 
overflow and with maximum performance.

See also: [`VisitationFrequency`](@ref), [`RectangularBinning`](@ref).

# Example 

```julia
using Entropies, DelayEmbeddings
D = Dataset(rand(100, 3))

# How shall the data be partitioned? 
# Here, we subdivide each coordinate axis into 4 equal pieces
# over the range of the data, resulting in rectangular boxes/bins
ϵ = RectangularBinning(4)

# Feed partitioning instructions to estimator.
est = VisitationFrequency(ϵ)

# Estimate a probability distribution over the partition
probabilities(D, est)
```
"""
function probabilities(x::Dataset, est::VisitationFrequency)
    non0hist(x, est.binning, normalize = true)
end

"""
# Visitation frequency, binning based entropy 

    genentropy(x::Dataset, est::VisitationFrequency, α::Real = 1; base::Real = 2)   

Compute the order-`α` generalized (Rényi) entropy[^Rényi1960] of a multivariate dataset `x`
using a visitation frequency approach.

## Description 

First, the state space defined by `x` is partitioned into rectangular boxes according to 
the binning instructions given by `est.binning`. Then, a histogram of visitations to 
each of those boxes is obtained, which is then sum-normalized to obtain a probability 
distribution, using [`probabilities`](@ref). The generalized entropy to the given `base` is 
then computed over that box visitation distribution using 
[`genentropy(::Real, ::AbstractArray)`](@ref).

## Example

```julia
using DelayEmbeddings, Entropies
D = Dataset(rand(1:3, 20000, 3))

# Estimator specification. Split each coordinate axis in five equal segments.
est = VisitationFrequency(RectangularBinning(5)) 

# Estimate order-1 (default) generalized entropy
Entropies.genentropy(D, est, base = 2)
```

See also: [`VisitationFrequency`](@ref).
"""
function genentropy(x::Dataset, est::VisitationFrequency, α::Real = 1; base::Real = 2)
    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))

    ps = probabilities(x, est)
    genentropy(α, ps, base = base)
end