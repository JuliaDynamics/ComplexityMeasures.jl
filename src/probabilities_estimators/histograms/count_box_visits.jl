import DelayEmbeddings: AbstractDataset, minima, maxima
import StaticArrays: SVector, MVector

"""
    encode_as_bin(point, reference_point, edgelengths) → Vector{Int}

Encode a point into its integer bin labels relative to some `reference_point`
(always counting from lowest to highest magnitudes), given a set of box
`edgelengths` (one for each axis). The first bin on the positive side of
the reference point is indexed with 0, and the first bin on the negative
side of the reference point is indexed with -1.

See also: [`joint_visits`](@ref), [`marginal_visits`](@ref).

## Example

```julia
using Entropies

refpoint = [0, 0, 0]
steps = [0.2, 0.2, 0.3]
encode_as_bin(rand(3), refpoint, steps)
```
"""
function encode_as_bin end

# TODO: All of these methods have exactly the same source code, why not just use Union?
# Or literally use AbstractVector{T}...
function encode_as_bin(point::Vector{T}, reference_point, edgelengths) where {T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode_as_bin(point::SVector{dim, T}, reference_point, edgelengths) where {dim, T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode_as_bin(point::MVector{dim, T}, reference_point, edgelengths) where {dim, T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode_as_bin(points::Vector{T}, reference_point, edgelengths) where {T <: Union{Vector, SVector, MVector}}
    [encode_as_bin(points[i], reference_point, edgelengths) for i = 1:length(points)]
end

function encode_as_bin(points::AbstractDataset{dim, T}, reference_point, edgelengths) where {dim, T}
    [encode_as_bin(points[i], reference_point, edgelengths) for i = 1:length(points)]
end


"""
    joint_visits(points, binning_scheme::RectangularBinning) → Vector{Vector{Int}}

Determine which bins are visited by `points` given the rectangular binning
scheme `ϵ`. Bins are referenced relative to the axis minima, and are
encoded as integers, such that each box in the binning is assigned a
unique integer array (one element for each dimension).

For example, if a bin is visited three times, then the corresponding
integer array will appear three times in the array returned.

See also: [`marginal_visits`](@ref), [`encode_as_bin`](@ref).

# Example

```julia
using DelayEmbeddings, Entropies

pts = Dataset([rand(5) for i = 1:100]);
joint_visits(pts, RectangularBinning(0.2))
```
"""
function joint_visits(points, binning_scheme::RectangularBinning)
    axis_minima, box_edge_lengths = minima_and_edgelengths(points, binning_scheme)
    # encode points relative to axis minima, with boxes of fixed length
    encode_as_bin(points, axis_minima, box_edge_lengths)
end

"""
    marginal_visits(points, binning_scheme::RectangularBinning, dims) → Vector{Vector{Int}}

Determine which bins are visited by `points` given the rectangular binning
scheme `ϵ`, but only along the desired dimensions `dims`. Bins are referenced
relative to the axis minima, and are encoded as integers, such that each box
in the binning is assigned a unique integer array (one element for each
dimension in `dims`).

For example, if a bin is visited three times, then the corresponding
integer array will appear three times in the array returned.

See also: [`joint_visits`](@ref), [`encode_as_bin`](@ref).

# Example

```julia
using DelayEmbeddings, Entropies
pts = Dataset([rand(5) for i = 1:100]);

# Marginal visits along dimension 3 and 5
marginal_visits(pts, RectangularBinning(0.3), [3, 5])

# Marginal visits along dimension 2 through 5
marginal_visits(pts, RectangularBinning(0.3), 2:5)
```
"""
function marginal_visits(points, binning_scheme::RectangularBinning, dims)
    axis_minima, box_edge_lengths = minima_and_edgelengths(points, binning_scheme)
    dim = length(axis_minima)
    if length(dims) == 1
        dims = [dim]
    end
    if sort(collect(dims)) == sort(collect(1:dim))
        joint_visits(points, binning_scheme)
    else
        # encode point relative to axis minima, with boxes of fixed length
        [encode_as_bin(pt, axis_minima, box_edge_lengths)[dims] for pt in points]
    end
end

"""
    marginal_visits(joint_visits, dims) → Vector{Vector{Int}}

If joint visits have been precomputed using [`joint_visits`](@ref), marginal
visits can be returned directly without providing the binning again
using the `marginal_visits(joint_visits, dims)` signature.

See also: [`joint_visits`](@ref), [`encode_as_bin`](@ref).

# Example

```
using DelayEmbeddings, Entropies
pts = Dataset([rand(5) for i = 1:100]);

# First compute joint visits, then marginal visits along dimensions 1 and 4
jv = joint_visits(pts, RectangularBinning(0.2))
marginal_visits(jv, [1, 4])

# Marginals along dimension 2
marginal_visits(jv, 2)
```
"""
function marginal_visits(joint_visits, dims)
    [bin[dims] for bin in joint_visits]
end