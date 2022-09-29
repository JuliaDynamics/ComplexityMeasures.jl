export RectangularBinning

import DelayEmbeddings: Dataset, minima, maxima
import StaticArrays: SVector, MVector

"""
    RectangularBinning(ϵ) <: AbstractBinning

Instructions for creating a rectangular box partition using the binning scheme `ϵ`.
Binning instructions are deduced from the type of `ϵ` as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals,
    extending the upper bound 1/100th of a bin size to ensure all points are covered.
2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting
    from the axis minima until the data is completely covered by boxes.
3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length
    intervals, extending the upper bound 1/100th of a bin size to ensure all points are
    covered.
4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size
    `ϵ[i]`, starting from the axis minima until the data is completely covered by boxes.
"""
struct RectangularBinning{E} <: AbstractBinning
    ϵ::E
end

# Extend this so that we use same function for vector or dataset input
DelayEmbeddings.minima(x::AbstractVector{<:Real}) = minimum(x)

"""
    get_minima_and_edgelengths(points,
        binning_scheme::RectangularBinning) → (Vector{Float}, Vector{Float})

Find the minima along each axis of the embedding, and computes appropriate
edge lengths given a rectangular `binning_scheme`, which provide instructions on how to
grid the space. Assumes the input is a vector of points.
"""
function get_minima_and_edgelengths(points, binning_scheme::RectangularBinning)
    ϵ = binning_scheme.ϵ

    D = length(points[1])
    n_pts = length(points)

    # TODO: This is INSANE, and the least performant thing on the universe. MUST FIX.
    axisminima = minimum.([minimum.([pt[i] for pt in points]) for i = 1:D])
    axismaxima = maximum.([maximum.([pt[i] for pt in points]) for i = 1:D])

    edgelengths = Vector{Float64}(undef, D)

    # Dictated by data ranges
    if ϵ isa Float64
        edgelengths = [ϵ for i in 1:D]
    elseif ϵ isa Vector{<:AbstractFloat}
        edgelengths .= ϵ
    elseif ϵ isa Int
        edgeslengths_nonadjusted = (axismaxima  - axisminima) / ϵ
        edgelengths = ((axismaxima + (edgeslengths_nonadjusted ./ 100)) - axisminima) ./ ϵ
    elseif ϵ isa Vector{Int}
        edgeslengths_nonadjusted = (axismaxima  .- axisminima) ./ ϵ
        edgelengths = ((axismaxima .+ (edgeslengths_nonadjusted ./ 100)) .- axisminima) ./ ϵ

    # Custom data ranges
    elseif ϵ isa Tuple{Vector{Tuple{T1, T2}},Int64} where {T1 <: Real, T2 <: Real}
        ranges = ϵ[1]
        n_bins = ϵ[2]
        length(ranges) == D || error("Tried to apply a $(length(ranges))-dimensional binning scheme to $(D)-dimensional data. Please provide ranges for all $(D) dimensions.")

        # We have predefined axis minima and axis maxima.
        stepsizes = zeros(Float64, D)
        edgelengths = zeros(Float64, D)
        for i = 1:D
            ranges[i][1] < ranges[i][2] || error("Range along dimension $i is not given as (minimum, maximum), got $(ranges[i])")
            edgelengths[i] = (maximum(ranges[i]) - minimum(ranges[i]))/n_bins
            axisminima[i] = minimum(ranges[i])
        end
    else
       error("get_minima_and_edgelengths not implemented for RectangularBinning(ϵ) with ϵ of type $(typeof(ϵ))")
    end

    axisminima, edgelengths
end

"""
    get_edgelengths(pts, binning_scheme::RectangularBinning) → Vector{Float}

Return the box edge length along each axis resulting from discretizing `pts` on a
rectangular grid specified by `binning_scheme`.

# Example

```julia
using Entropies, DelayEmbeddings
pts = Dataset([rand(5) for i = 1:1000])

get_edgelengths(pts, RectangularBinning(0.6))
get_edgelengths(pts, RectangularBinning([0.5, 0.3, 0.3, 0.4, 0.4]))
get_edgelengths(pts, RectangularBinning(8))
get_edgelengths(pts, RectangularBinning([10, 8, 5, 4, 22]))
```
"""
function get_edgelengths end

get_edgelengths(points, ϵ::RectangularBinning) = get_minima_and_edgelengths(points, ϵ)[2]

# need this for type stability when getting minima and edgelength for a Dataset for fasthist
# otherwise, we're subtracting and dividing static vectors with regular vectors, which allocates
# all over the place.
import StaticArrays: SVector

function minima_edgelengths(points::AbstractDataset{D, T}, binning_scheme::RectangularBinning) where {D, T<:Real}
    mini, edgelengths = get_minima_and_edgelengths(points, binning_scheme)
    return SVector{D, T}(mini), SVector{D, Float64}(float.(edgelengths))
end
get_edgelengths(points::AbstractDataset, ϵ::RectangularBinning) = minima_edgelengths(points, ϵ)[2]



#=
TODO: for @kahaaga
This insanely complicated, and difficult to reason for method, should be a different
binning scheme if you really need it.


5. `ϵ::Tuple{Vector{Tuple{Float64,Float64}}, Int64}` (see below for example) creates
    intervals along each coordinate axis from ranges indicated by a vector of `(min, max)`
    tuples, then divides each coordinate axis into an integer number of equal-length
    intervals. *Note: this does not ensure
    that all points are covered by the data (points outside the binning are ignored)*.

# Example 2: Custom grids (partition not guaranteed to cover data points):

Assume the data consists of 3-dimensional points `(x, y, z)`, and that we want a grid
that is fixed over the intervals `[x₁, x₂]` for the first dimension, over `[y₁, y₂]`
for the second dimension, and over `[z₁, z₂]` for the third dimension. We when want to
split each of those ranges into 4 equal-length pieces. *Beware: some points may fall
outside the partition if the intervals are not chosen properly (these points are
simply discarded)*.

The following binning specification produces the desired (hyper-)rectangular boxes.

```julia
using Entropies, DelayEmbeddings

D = Dataset(rand(100, 3));

x₁, x₂ = 0.5, 1 # not completely covering the data, which are on [0, 1]
y₁, y₂ = -2, 1.5 # covering the data, which are on [0, 1]
z₁, z₂ = 0, 0.5 # not completely covering the data, which are on [0, 1]

ϵ = [(x₁, x₂), (y₁, y₂), (z₁, z₂)], 4 # [interval 1, interval 2, ...], n_subdivisions

RectangularBinning(ϵ)
```
=#