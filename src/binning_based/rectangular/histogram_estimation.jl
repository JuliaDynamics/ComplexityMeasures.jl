export genentropy

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
