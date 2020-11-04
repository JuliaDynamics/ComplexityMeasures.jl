export non0hist, binhist

import DelayEmbeddings: AbstractDataset

""" 
# Histograms from collections

    non0hist(x::AbstractVector; normalize::Bool = true) → p::Vector{Float64}

Compute the unordered histogram of the values of `x`, directly from the distribution of 
values, without any coarse-graining or discretization. Assumes that `x` can be sorted.

If `normalize==true`, then the 
histogram is sum-normalized. If `normalize==false`, then occurrence counts for the unique 
elements in `x` is returned. 

## Example 

```julia
using Entropies
x = rand(1:10, 100000)
Entropies.non0hist(x) # sum-normalized
Entropies.non0hist(x, normalize = false) # histogram (counts)
```

# Histograms of `Dataset`s

    non0hist(x::Dataset; normalize::Bool = true) → p::Vector{Float64}

Compute the unordered histogram of the values of the `Dataset` `x` , directly from the 
distribution of points, without any coarse-graining or discretization.

## Example 

```julia
using DelayEmbeddings, Entropies
D = Dataset(rand(1:3, 50000, 3))
Entropies.non0hist(D) # sum-normalized
Entropies.non0hist(D, normalize = false) # histogram (counts)
```
"""
function non0hist(x::AbstractVector{T}; normalize::Bool = true) where T<:Real
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
    if normalize 
        return hist ./ L
    else 
        return hist
    end
end

# The following is from chaostools
##################################
"""
    non0hist(ε, dataset::AbstractDataset; normalize = true) → p

Partition a dataset into tabulated intervals (boxes) of
size `ε` and return the (sum-normalized, if `normalize==true`) histogram in an 
unordered 1D form, discarding all zero elements and bin edge information.

## Performances Notes
This method has a linearithmic time complexity (`n log(n)` for `n = length(data)`)
and a linear space complexity (`l` for `l = dimension(data)`).
This allows computation of histograms of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

Use [`binhist`](@ref) to retain bin edge information.
"""
non0hist(args...) = _non0hist(args...)[1]


function _non0hist(ε::Real, data::AbstractDataset{D, T}; normalize::Bool = true) where {D, T<:Real}
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

    if normalize 
        return hist ./ L, bins, mini
    else 
        return hist, bins, mini
    end
end

"""
    binhist(ε, data; normalize = true) → p, bins
Do the same as [`non0hist`](@ref) but also return the bin edge information.
"""
function binhist(ε, data; normalize::Bool = true)
    hist, bins, mini = _non0hist(ε, data, normalize = normalize)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
