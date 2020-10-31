export non0hist, binhist

import DelayEmbeddings: AbstractDataset

""" 
    non0hist(x::AbstractVector{<:Real}) → p::Vector{Float64}

Compute the sum-normalized unordered histogram of the values of `x`. Assumes `x` can be 
sorted.
"""
function non0hist(x::AbstractVector{T}) where T<:Real
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
            push!(hist, count/L)
            prev_val = val
            count = 1
        end
    end
    push!(hist, count/L)

    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return hist
end

# The following is from chaostools
##################################
"""
    non0hist(ε, dataset::AbstractDataset) → p

Partition a dataset into tabulated intervals (boxes) of
size `ε` and return the sum-normalized histogram in an unordered 1D form,
discarding all zero elements and bin edge information.

## Performances Notes
This method has a linearithmic time complexity (`n log(n)` for `n = length(data)`)
and a linear space complexity (`l` for `l = dimension(data)`).
This allows computation of histograms of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

Use [`binhist`](@ref) to retain bin edge information.
"""
non0hist(args...) = _non0hist(args...)[1]


function _non0hist(ε::Real, data::AbstractDataset{D, T}) where {D, T<:Real}
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
            push!(hist, count/L)
            prev_bin = bin
            count = 1
        end
    end
    push!(hist, count/L)

    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return hist, bins, mini
end

"""
    binhist(ε, data) → p, bins
Do the same as [`non0hist`](@ref) but also return the bin edge information.
"""
function binhist(ε, data)
    hist, bins, mini = _non0hist(ε, data)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
