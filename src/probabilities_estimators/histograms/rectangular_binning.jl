export RectangularBinning

"""
    RectangularBinning(ϵ) <: AbstractBinning

Rectangular box partition of state space using the scheme `ϵ`.
Binning instructions are deduced from the type of `ϵ` as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals
    that cover all data.
2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting
    from the axis minima until the data is completely covered by boxes.
3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length
    intervals that cover all data.
4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size
    `ϵ[i]`, starting from the axis minima until the data is completely covered by boxes.
"""
struct RectangularBinning{E} <: AbstractBinning
    ϵ::E
end

function probabilities(x::Vector_or_Dataset, binning::RectangularBinning)
    fasthist(x, binning)[1]
end
function probabilities(x::Vector_or_Dataset, ε::Union{Real, Vector{<:Real}})
    probabilities(x, RectangularBinning(ε))
end


"""
    bin_encoder(x, binning_scheme::RectangularBinning)

Find the minima along each dimension, and compute appropriate
edge lengths for each dimension of `x` given a rectangular binning.
Put them in a `RectangularBinEncoder` that can be then used to map points into bins.
"""
function bin_encoder(x::AbstractDataset{D,T}, b::RectangularBinning) where {D, T}
    # This function always returns static vectors and is type stable
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    v = ones(SVector{D,T})
    if ϵ isa Float64 || ϵ isa Vector{<:AbstractFloat}
        edgelengths = ϵ .* v
    elseif ϵ isa Int || ϵ isa Vector{Int}
        edgeslengths_nonadjusted = @. (maxi - mini)/ϵ
        # just taking the next float here is enough to ensure boxes cover data
        edgelengths = nextfloat.(edgeslengths_nonadjusted)
    end
    RectangularBinEncoder(mini, edgelengths)
end

function bin_encoder(x::AbstractVector{<:Real}, b::RectangularBinning)
    # This function always returns numbers and is type stable
    ϵ = b.ϵ
    mini, maxi = extrema(x)
    if ϵ isa AbstractFloat
        edgelength = ϵ
    elseif ϵ isa Int
        edgeslength_nonadjusted = (maxi - mini)/ϵ
        # just taking the next float here should be enough
        edgelength = nextfloat(edgeslength_nonadjusted)
        # edgelengths = ((axismaxima + (edgeslengths_nonadjusted ./ 100)) - axisminima) ./ ϵ
    else
        error("Invalid ϵ for binning of a vector")
    end
    RectangularBinEncoder(mini, edgelength)
end

struct RectangularBinEncoder{M, E} <: AbstractBinEncoder
    mini::M # fields are either static vectors or numbers
    edgelengths::E
end

# This function is the same as `symbolize`.
function encode_as_bins(x::Vector_or_Dataset, b::RectangularBinEncoder)
    (; mini, edgelengths) = b
    # Map each datapoint to its bin edge (hence, we are symbolizing the x here)
    # (notice that this also works for vector x, and broadcasting is ignored)
    bins = map(point -> floor.(Int, (point .- mini) ./ edgelengths), x)
    return bins
end
function symbolize(x::Vector_or_Dataset, b::RectangularBinEncoder)
    return encode_as_bins(x, b)
end


###########################################################################################
# OLD CODE! Also terrible performance, no static vectors!!! Will be deleted!
###########################################################################################
function minima_and_edgelengths_OLD(points, binning_scheme::RectangularBinning)
    # TODO: Ensure this function returns static vectors!!!!!!!!!!!
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
       error("minima_and_edgelengths not implemented for RectangularBinning(ϵ) with ϵ of type $(typeof(ϵ))")
    end

    axisminima, edgelengths
end

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