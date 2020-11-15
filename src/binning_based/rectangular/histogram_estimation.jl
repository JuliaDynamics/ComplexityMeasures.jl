export binhist

"""
    _non0hist(points, binning_scheme::RectangularBinning, dims)

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
h1 = _non0hist(pts, RectangularBinning(0.2), 1:3)
h2 = _non0hist(pts, RectangularBinning(0.2), [1, 2])

# Test that we're actually getting normalised histograms
sum(h1) ≈ 1.0, sum(h2) ≈ 1.0
```
"""
function _non0hist(points, binning_scheme::RectangularBinning, dims)
    bin_visits = marginal_visits(points, binning_scheme, dims)
    _non0hist(bin_visits)
end

function _non0hist(points::AbstractDataset{N, T}, binning_scheme::RectangularBinning) where {N, T}
    _non0hist(points, binning_scheme, 1:N)
end


# TODO: ϵ::RectangularBinning(Float64) allocates slightly more memory than the method below
# because of the call to `minima_edgelengths`. Can be optimized, but keep both versions for
# now and merge docstrings.
probabilities(data::AbstractDataset, ϵ::RectangularBinning) = _non0hist(data, ϵ)[1]
function _non0hist(data::AbstractDataset{D, T}, ϵ::RectangularBinning) where {D, T<:Real}

    # TODO: this allocates a lot, but is not performance critical
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

function binhist(x::AbstractDataset{D, T}, ϵ::RectangularBinning) where {D, T<:Real}
    hist, bins, mini, edgelengths = _non0hist(x, ϵ)
    unique!(bins)
    b = [β .* edgelengths .+ mini for β in bins]
    return hist, b
end

# The following is originally from ChaosTools.jl
probabilities(data::AbstractDataset, ε::Real) = _non0hist(data, ε)[1]
function _non0hist(data::AbstractDataset{D, T}, ε::Real) where {D, T<:Real}
    mini = minima(data)
    L = length(data)
    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point - mini)/ε), data)
    sort!(bins, alg=QuickSort)

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
    binhist(x::Dataset, ε::Real) → p, bins
    binhist(x::Dataset, ε::RectangularBinning) → p, bins

Hyper-optimized histogram calculation for `x` with rectangular binning `ε`.
Returns the probabilities `p` of each bin of the histogram as well as the bins.
Notice that `bins` are the starting corners of each bin. If `ε isa Real`, then the actual
bin size is `ε` across each dimension. If `ε isa RectangularBinning`, then the bin
size for each dimension will depend on the binning scheme.

See also: [`RectangularBinning`](@ref).
"""
function binhist(data, ε)
    hist, bins, mini = _non0hist(data, ε)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
