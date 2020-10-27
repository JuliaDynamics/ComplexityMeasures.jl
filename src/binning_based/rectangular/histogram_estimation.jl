export non0hist

"""
    _non0hist(bin_visits::Vector{T}) where {T <: Union{Vector, SVector, MVector}} → Vector{Real}
    _non0hist(x::Dataset) → Vector{Real}

Return the unordered histogram (visitation frequency) over the array of `bin_visits`,
which is a vector containing bin encodings (each point encoded by an integer vector).

# Example

```julia 
using Entropies, DelayEmbeddings
pts = Dataset([rand(5) for i = 1:100]);

# Histograms from precomputed joint/marginal visitations 
jv = joint_visits(pts, RectangularBinning(10))
mv = marginal_visits(pts, RectangularBinning(10), 1:3)

h1 = non0hist(jv)
h2 = non0hist(mv)

# Test that we're actually getting a normalised histograms
sum(h1) ≈ 1.0, sum(h2) ≈ 1.0
```
"""
function non0hist(bin_visits::Vector{T}) where {T <: Union{Vector, SVector, MVector}}
    L = length(bin_visits)
    hist = Vector{Float64}()

    # Reserve enough space for histogram:
    sizehint!(hist, L)

    sort!(bin_visits, alg = QuickSort)

    # Fill the histogram by counting consecutive equal bins:
    prev_bin = bin_visits[1]
    count = 1
    @inbounds for i in 2:L
        bin = bin_visits[i]
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

    return hist
end

non0hist(x::Dataset) = non0hist(x.data)


"""
    non0hist(points, binning_scheme::RectangularBinning, dims)

Determine which bins are visited by `points` given the rectangular `binning_scheme`, 
considering only the marginal along dimensions `dims`. Bins are referenced 
relative to the axis minima.

Returns the unordered histogram (visitation frequency) over the array of bin visits.

# Example 
```julia 
using using Entropies, DelayEmbeddings
pts = Dataset([rand(5) for i = 1:100]);

# Histograms directly from points given a rectangular binning scheme
h1 = non0hist(pts, RectangularBinning(0.2), 1:3) 
h2 = non0hist(pts, RectangularBinning(0.2), [1, 2])

# Test that we're actually getting normalised histograms 
sum(h1) ≈ 1.0, sum(h2) ≈ 1.0
```
"""
function non0hist(points, binning_scheme::RectangularBinning, dims)
    bin_visits = marginal_visits(points, binning_scheme, dims)
    L = length(bin_visits)
    hist = Vector{Float64}()

    # Reserve enough space for histogram:
    sizehint!(hist, L)

    sort!(bin_visits, alg = QuickSort)

    # Fill the histogram by counting consecutive equal bins:
    prev_bin = bin_visits[1]
    count = 1
    @inbounds for i in 2:L
        bin = bin_visits[i]
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

    return hist
end

function non0hist(points::Dataset{N, T}, binning_scheme::RectangularBinning) where {N, T}
    non0hist(points, binning_scheme, 1:N)
end