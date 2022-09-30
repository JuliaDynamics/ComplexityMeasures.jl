# Notice that in the new `event` API, there is no reason to export the `fasthist`
# function, because it is just the `probabilities` call,
# while the old `binhist` function becomes the dispatch of `event`
# for `RectangularBinning`. Deprecations will be added for `binhist` of course.

# Originally this code was in ChaosTools.jl, many years ago. It has been transferred
# here, and then expanded to work with different binning configurations.
# It also has been made generic to work with arbitrary inputs without code duplication.

# Internal function docstring
"""
    fasthist(x::Vector_or_Dataset, binning::AbstractBinning)

Hyper-optimized histogram calculation for `x` with rectangular binning.
Return the probabilities `p` of each bin of the histogram, the bins
(in integer coordinates), and the encoder, that can map points into bins.

For rectangular binning this method has a linearithmic time complexity
(`n log(n)` for `n = length(x)`) and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of histograms of high-dimensional
datasets and with small bin sizes without memory overflow and with maximum performance.
"""
function fasthist(x::Vector_or_Dataset, ϵ::AbstractBinning)
    encoder = bin_encoder(x, ϵ)
    bins = encode_as_bins(x, encoder)
    hist = fasthist(bins)
    return Probabilities(hist), bins, encoder
end

"""
    fasthist(x) → c::Vector{Int}

Count the occurrences `c` of the unique data values in `x`.
Return them as raw data, i.e., `Vector{Int}`.

Useful mostly when `x` contains integer or categorical data.
The actual values the counts correspond to are `sort!(unique(x))`, but are not
returned.

This function works for any `x` for which `sort!(x)` works.
So, it also mutates `x`. That's why it's called with `copy` in higher level
API when necessary.
"""
function fasthist(x)
    L = length(x)
    hist = Vector{Int}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)
    # Fill the histogram by counting consecutive equal values:
    sort!(x; alg = QuickSort)
    prev_val, count = x[1], 0
    for val in x
        if val == prev_val
            count += 1
        else
            push!(hist, count)
            prev_val = val
            count = 1
        end
    end
    push!(hist, count)
    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return hist
end

###########################################################################################
# Old code
###########################################################################################
"""
    fasthist(points, binning_scheme::RectangularBinning, dims)

Determine which bins are visited by `points` given the rectangular `binning_scheme`,
considering only the marginal along dimensions `dims`. Bins are referenced
relative to the axis minima.

Returns the unordered (sum-normalized, if `normalize==true`) histogram (visitation
frequency) over the array of bin visits.

# Example
```julia
using using Entropies, DelayEmbeddings
pts = Dataset([rand(5) for i = 1:100]);

# Histograms directly from points given a rectangular binning scheme
h1 = fasthist(pts, RectangularBinning(0.2), 1:3)
h2 = fasthist(pts, RectangularBinning(0.2), [1, 2])

# Test that we're actually getting normalised histograms
sum(h1) ≈ 1.0, sum(h2) ≈ 1.0
```
"""
function fasthist_OLD(points, binning_scheme::RectangularBinning, dims)
    bin_visits = marginal_visits(points, binning_scheme, dims)
    fasthist(bin_visits)
end

function fasthist_OLD(points::AbstractDataset{N, T}, binning_scheme::RectangularBinning) where {N, T}
    fasthist(points, binning_scheme, 1:N)
end

"""
    binhist(x::AbstractDataset, ε::Real) → p, bins
    binhist(x::AbstractDataset, ε::RectangularBinning) → p, bins

Hyper-optimized histogram calculation for `x` with rectangular binning `ε`.
Returns the probabilities `p` of each bin of the histogram as well as the bins.
Notice that `bins` are the starting corners of each bin. If `ε isa Real`, then the actual
bin size is `ε` across each dimension. If `ε isa RectangularBinning`, then the bin
size for each dimension will depend on the binning scheme.

See also: [`RectangularBinning`](@ref).
"""
function binhist(x, ϵ::RectangularBinning)
    hist, bins, mini, edgelengths = fasthist(x, ϵ)
    unique!(bins)
    b = [β .* edgelengths .+ mini for β in bins]
    return hist, b
end
