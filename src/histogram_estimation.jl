export binhist

probabilities(x) = _non0hist(x)
function _non0hist(x)
    L = length(x)

    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)

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
_non0hist(x::AbstractDataset) = _non0hist(x.data)
probabilities(x::AbstractDataset) = _non0hist(x.data)


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
Hyper-optimized histogram calculation for `x` with rectangular binning `ε`.
Returns the probabilities `p` of each bin of the histogram as well as the bins.
Notice that `bins` are the starting corners of each bin. The actual bin size is
`ε` across each dimension.
"""
function binhist(data, ε)
    hist, bins, mini = _non0hist(data, ε)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
