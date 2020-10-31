export VisitationFrequency, entropy, probabilities, genentropy
import DelayEmbeddings: Dataset, AbstractDataset

"""
    VisitationFreqency(r::RectangularBinning; b::Real = 2)

A probability estimator based on binning data into rectangular boxes dictated by 
the binning scheme `r`.

If the estimator is used for entropy computation, then the entropy is computed 
to base `b` (the default `b = 2` gives the entropy in bits).

See also: [`RectangularBinning`](@ref).
"""
struct VisitationFrequency <: BinningProbabilitiesEstimator
    binning::RectangularBinning
    b::Real
    
    function VisitationFrequency(r::RectangularBinning; b::Real = 2)
        new(r, b)
    end
end

"""
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
    non0hist(x, est.binning)
end

"""
    entropy(x::Dataset, est::VisitationFrequency, α::Real = 1) → Real


Estimate the generalized order `α` entropy of `x` using a visitation frequency approach. 
This is done by first estimating the sum-normalized unordered 1D histogram using
[`probabilities`](@ref), then computing entropy over that histogram/distribution.

The base `b` of the logarithms is inferred from the provided estimator 
(e.g. `est = VisitationFrequency(RectangularBinning(45), b = Base.MathConstants.e`).

## Description

Let ``p`` be an array of probabilities (summing to 1). Then the Rényi entropy is

```math
H_\\alpha(p) = \\frac{1}{1-\\alpha} \\log \\left(\\sum_i p[i]^\\alpha\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [^Shannon1948]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)

See also: [`VisitationFrequency`](@ref), [`RectangularBinning`](@ref).

## Example

```julia
using Entropies, DelayEmbeddings
D = Dataset(rand(100, 3))

# How shall the data be partitioned? Here, we subdivide each 
# coordinate axis into 4 equal pieces over the range of the data, 
# resulting in rectangular boxes/bins (see RectangularBinning).
ϵ = RectangularBinning(4)

# Estimate entropy
entropy(D, VisitationFrequency(ϵ))
```
"""
function entropy(x::Dataset, est::VisitationFrequency, α::Real = 1)
    ps = probabilities(x, est)

    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))
    if α ≈ 0 # Hartley entropy, max-entropy
        return log(est.b, length(ps)) 
    elseif α ≈ 1
        return -sum( x*log(est.b, x) for x in ps ) #Shannon entropy
    elseif isinf(α)
        return -log(est.b, maximum(ps)) #Min entropy
    else
        return (1/(1-α))*log(est.b, sum(x^α for x in ps) ) #Renyi α entropy
    end
end


"""
    genentropy(α::Real, x::Dataset, est::VisitationFrequency) where {N, T <: Real}

Compute the `α` order generalized (Rényi) entropy[^Rényi1960] of a multivariate dataset `x`.

## Description 

First, the state space defined by `x` is partitioned into rectangular boxes according to 
the binning instructions given by `est.binning`. Then, a histogram of visitations to 
each of those boxes is obtained, which is then sum-normalized to obtain a probability 
distribution, using [`probabilities`](@ref). The generalized entropy to base `est.b` is 
then computed over that box visitation distribution using 
[`genentropy(::Real, ::AbstractArray)`](@ref).

See also: [`VisitationFrequency`](@ref).
"""
function genentropy(α::Real, x::Dataset{N, T}, est::VisitationFrequency) where {N, T <: Real}
    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))

    ps = probabilities(x, est)
    genentropy(α, ps, base = est.b)
end