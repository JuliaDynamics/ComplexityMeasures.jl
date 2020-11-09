export non0hist, binhist

import DelayEmbeddings: AbstractDataset

"""
    non0hist(x::Vector_or_Dataset) → p::Probabilities

Compute the unordered histogram of the values of `x`, directly from the distribution of
values, without any coarse-graining or discretization. Assumes that `x` can be sorted.

    non0hist(dataset::Vector_or_Dataset, ε::Real) → p::Probabilities

Partition a dataset into tabulated intervals (boxes) of
size `ε` and return the sum-normalized histogram in an
unordered 1D form, discarding all zero elements and bin edge information.
Use [`binhist`](@ref) to retain bin edge information.

This method has a linearithmic time complexity (`n log(n)` for `n = length(data)`)
and a linear space complexity (`l` for `l = dimension(data)`).
This allows computation of histograms of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.

Notice that `non0hist` returns probabilities. To obtain counts simply do
`counts = p * length(x)`.

## Example

```julia
using DelayEmbeddings, Entropies
A = Dataset(rand(1:3, 50000, 3))
non0hist(A)
# discretizing version
B = Dataset(rand(1000, 3))
non0hist(B, 0.01)
```
"""
function non0hist(x::AbstractVector{T}) where T
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

# The following is originally from ChaosTools.jl
non0hist(data::AbstractDataset, ε::Real) = _non0hist(data, ε)[1]
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

"""
    binhist(data::Dataset, ε::Real; normalize = true) → p, bins
Do the same as [`non0hist`](@ref) but also return the bin edge information.
"""
function binhist(data, ε)
    hist, bins, mini = _non0hist(ε, data)
    unique!(bins)
    b = [β .* ε .+ mini for β in bins]
    return hist, b
end
