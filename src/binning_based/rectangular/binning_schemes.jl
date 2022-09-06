export RectangularBinning

import DelayEmbeddings: Dataset, minima, maxima
import StaticArrays: SVector, MVector

""" 
    BinningScheme

The supertype of all binning schemes in the CausalityTools ecosystem. 
"""
abstract type BinningScheme end

""" 
    RectangularBinningScheme <: BinningScheme

The supertype of all rectangular binning schemes in the CausalityTools ecosystem.
""" 
abstract type RectangularBinningScheme <: BinningScheme end

"""
    RectangularBinning(ϵ) <: RectangularBinningScheme
    
Instructions for creating a rectangular box partition using the binning scheme `ϵ`. 
Binning instructions are deduced from the type of `ϵ`.


Rectangular binnings may be automatically adjusted to the data in which the `RectangularBinning` 
is applied, as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals, 
    extending the upper bound 1/100th of a bin size to ensure all points are covered.

2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting 
    from the axis minima until the data is completely covered by boxes.

3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length 
    intervals, extending the upper bound 1/100th of a bin size to ensure all points are 
    covered.

4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size `ϵ[i]`, starting 
    from the axis minima until the data is completely covered by boxes.

Rectangular binnings may also be specified on arbitrary min-max ranges. 

5. `ϵ::Tuple{Vector{Tuple{Float64,Float64}},Int64}` creates intervals 
    along each coordinate axis from ranges indicated by a vector of `(min, max)` tuples, then divides 
    each coordinate axis into an integer number of equal-length intervals. *Note: this does not ensure 
    that all points are covered by the data (points outside the binning are ignored)*.


# Example 1: Grid deduced automatically from data (partition guaranteed to cover data points)

## Flexible box sizes 

The following binning specification finds the minima/maxima along each coordinate axis, then 
split each of those data ranges (with some tiny padding on the edges) into `10` equal-length 
intervals. This gives (hyper-)rectangular boxes, and works for data of any dimension.

```julia
using Entropies
RectangularBinning(10)
```

Now, assume the data consists of 2-dimensional points, and that we want a finer grid along
one of the dimensions than over the other dimension.

The following binning specification finds the minima/maxima along each coordinate axis, then 
splits the range along the first coordinate axis (with some tiny padding on the edges) 
into `10` equal-length intervals, and the range along the second coordinate axis (with some 
tiny padding on the edges) into `5` equal-length intervals.
This gives (hyper-)rectangular boxes.

```julia
using Entropies
RectangularBinning([10, 5])
```

## Fixed box sizes 

The following binning specification finds the minima/maxima along each coordinate axis, 
then split the axis ranges into equal-length intervals of fixed size `0.5` until the all data 
points are covered by boxes. This approach yields (hyper-)cubic boxes, and works for 
data of any dimension.

```julia
using Entropies
RectangularBinning(0.5)
```

Again, assume the data consists of 2-dimensional points, and that we want a finer grid along
one of the dimensions than over the other dimension.

The following binning specification finds the minima/maxima along each coordinate axis, then splits
the range along the first coordinate axis into equal-length intervals of size `0.3`,
and the range along the second axis into equal-length intervals of size `0.1` (in both cases, 
making sure the data are completely covered by the boxes).
This approach gives a (hyper-)rectangular boxes. 

```julia
using Entropies
RectangularBinning([0.3, 0.1])
```

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
"""
struct RectangularBinning{E} <: RectangularBinningScheme
    ϵ::E
end

"""
    get_minima(pts) → SVector

Return the minima along each axis of the dataset `pts`.
"""
function get_minima end

"""
    get_maxima(pts) → SVector

Return the maxima along each axis of the dataset `pts`.
"""
function get_maxima end

"""
    get_minmaxes(pts) → Tuple{Vector{Float}, Vector{Float}}

Return a vector of tuples containing axis-wise (minimum, maximum) values.
"""
function get_minmaxes end

function get_minima(pts::AbstractDataset)
    minima(pts)
end

function get_minima(pts::Vector{T}) where {T <: Union{SVector, MVector, Vector}}
    minima(Dataset(pts))
end

function get_maxima(pts::AbstractDataset)
    maxima(pts)
end

function get_maxima(pts::Vector{T}) where {T <: Union{SVector, MVector, Vector}}
    maxima(Dataset(pts))
end


function get_minmaxes(pts::AbstractDataset)
    mini, maxi = minima(pts), maxima(pts)
    minmaxes = [(mini[i], maxi[i]) for i = 1:length(mini)]
end

function get_minmaxes(pts::Vector{T}) where {T <: Union{SVector, MVector, Vector}}
    get_minmaxes(Dataset(pts))
end


"""
    get_minima_and_edgelengths(points, 
        binning_scheme::RectangularBinning) → (Vector{Float}, Vector{Float})

Find the minima along each axis of the embedding, and computes appropriate
edge lengths given a rectangular `binning_scheme`, which provide instructions on how to 
grid the space. Assumes the input is a vector of points.

See documentation for [`RectangularBinning`](@ref) for details on the 
binning scheme.

# Example

## Example 1: Grid deduced automatically from data (partition guaranteed to cover data points)

```julia
using Entropies, DelayEmbeddings
pts = Dataset([rand(4) for i = 1:1000])

get_minima_and_edgelengths(pts, RectangularBinning(0.6))
get_minima_and_edgelengths(pts, RectangularBinning([0.5, 0.3, 0.4, 0.4]))
get_minima_and_edgelengths(pts, RectangularBinning(10))
get_minima_and_edgelengths(pts, RectangularBinning([10, 8, 5, 4]))
```

## Example 2: Custom grids (partition not guaranteed to cover data points): 

```julia
using Entropies, DelayEmbeddings

# Here, we split the first, second and third coordinate axes in 
# five equal-length pieces, defined over the intervals [a, b],
# [c, d], and [e, f], respectively (beware: some points may fall 
# outside the partition if the intervals are not chosen properly).
D = Dataset(rand(100, 3));
a, b = 0.5, 1 # not completely covering the data, which are on [0, 1]
c, d = -2, 1.5 # covering the data, which are on [0, 1]
e, f = 0, 0.5 # not completely covering the data, which are on [0, 1]
ϵ = [(a, b), (c, d), (e, f)], 3
bin = RectangularBinning(ϵ)
get_minima_and_edgelengths(D, bin)
```
"""
function get_minima_and_edgelengths(points, binning_scheme::RectangularBinning)
    ϵ = binning_scheme.ϵ

    D = length(points[1])
    n_pts = length(points)

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

# need this for type stability when getting minima and edgelength for a Dataset for _non0hist
# otherwise, we're subtracting and dividing static vectors with regular vectors, which allocates
# all over the place.
import StaticArrays: SVector

function minima_edgelengths(points::AbstractDataset{D, T}, binning_scheme::RectangularBinning) where {D, T<:Real}
    mini, edgelengths = get_minima_and_edgelengths(points, binning_scheme)
    return SVector{D, T}(mini), SVector{D, Float64}(float.(edgelengths))
end
get_edgelengths(points::AbstractDataset, ϵ::RectangularBinning) = minima_edgelengths(points, ϵ)[2]
