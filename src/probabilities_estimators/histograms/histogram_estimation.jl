# Notice that in the new `event` API, there is no reason to export the `fasthist`
# function, because it is just the `probabilities` call,
# while the old `binhist` function becomes the dispatch of `event`
# for `RectangularBinning`. Deprecations will be added for `binhist` of course.

# Originally this code was in ChaosTools.jl, many years ago. It has been transferred
# here, and then expanded to work with different binning configurations.

# Internal function docstring
"""
    fasthist(x::Vector_or_Dataset, binning::RectangularBinning)
    fasthist(x::Vector_or_Dataset, ε::Union{<:Real, <:Vector})

Hyper-optimized histogram calculation for `x` with rectangular binning.
Return the probabilities `p` of each bin of the histogram, the bins
(in integer coordinates), the minima of the histogram support, and the edge lengths.

Use `binhist` (TO BE RENAMED TO `events`) to get the bins in state space coordinates.

This method has a linearithmic time complexity (`n log(n)` for `n = length(x)`)
and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of histograms of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

See [`RectangularBinning`](@ref) for all possible binning configurations.
"""
function fasthist end

# Dataset implementation:
function fasthist(data::AbstractDataset{D, T}, ϵ::RectangularBinning) where {D, T<:Real}
    # TODO: this allocates a lot, but is not performance critical...?
    mini, edgelengths = minima_edgelengths(data, ϵ)
    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point .- mini) ./ edgelengths), data)
    sort!(bins, alg=QuickSort)
    # Reserve enough space for histogram:
    L = length(data)
    hist = Vector{Float64}()
    sizehint!(hist, L)
    # Fill the histogram by counting consecutive equal bins:
    prev_bin, count = bins[1], 0
    for bin in bins
        if bin == prev_bin
            count += 1
        else
            push!(hist, count)
            prev_bin = bin
            count = 1
        end
    end
    push!(hist, count)
    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return Probabilities(hist ./ L), bins, mini, edgelengths
end

# Vector implementation
# TODO: Paste here.


# Count occurrences implementation (direct counting of identical elements)

# Frequencies are needed elsewhere in the package too, so keep in its own method.
"""
    fasthist(x::Vector_or_Dataset)

Equivalent with `probabilities(x)` or with `probabilities(x, CountOccurrences)`.
See [`CountOccurrences`](@ref).
"""
function fasthist(x)
    L = length(x)
    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)
    # Sort
    sx = sort(x, alg = QuickSort)
    # Fill the histogram by counting consecutive equal values:
    prev_val, count = sx[1], 0
    for val in sx
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
    return Probabilities(hist ./ L)
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
function fasthist(points, binning_scheme::RectangularBinning, dims)
    bin_visits = marginal_visits(points, binning_scheme, dims)
    fasthist(bin_visits)
end

function fasthist(points::AbstractDataset{N, T}, binning_scheme::RectangularBinning) where {N, T}
    fasthist(points, binning_scheme, 1:N)
end


# TODO: ϵ::RectangularBinning(Float64) allocates slightly more memory than the method below
# because of the call to `minima_edgelengths`. Can be optimized, but keep both versions for
# now and merge docstrings.

function binhist(x::AbstractDataset{D, T}, ϵ::RectangularBinning) where {D, T<:Real}
    hist, bins, mini, edgelengths = fasthist(x, ϵ)
    unique!(bins)
    b = [β .* edgelengths .+ mini for β in bins]
    return hist, b
end

# Convenience dispatch for when an integer is given.
probabilities(data, n::Integer) =
probabilities(Dataset(data), VisitationFrequency(RectangularBinning(n)))

# Convenience dispatch for when a float is given.
probabilities(data, ε::AbstractFloat) = probabilities(Dataset(data), ε)

# The following is originally from ChaosTools.jl
probabilities(data::AbstractDataset, ε::AbstractFloat) = fasthist(data, ε)[1]
function fasthist(data::AbstractDataset{D, T}, ε::AbstractFloat) where {D, T<:Real}
    mini = minima(data)
    L = length(data)
    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point - mini)/ε), data)
    sort!(bins; alg=QuickSort)

    # Fill the histogram by counting consecutive equal bins:
    prev_bin, count = bins[1], 0
    for bin in bins
        if bin == prev_bin
            count += 1
        else
            push!(hist, count)
            prev_bin = bin
            count = 1
        end
    end
    push!(hist, count)

    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return Probabilities(hist ./ L), bins, mini
end

# TODO: This needs to be expanded to allow `est::RectangularBinning(vector)`
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
function binhist(data, ε)
    hist, bins, mini = fasthist(data, ε)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
