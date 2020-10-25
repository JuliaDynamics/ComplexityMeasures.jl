import DelayEmbeddings: Dataset, AbstractDataset, minima, maxima
import StaticArrays: SVector, MVector

export encode 

"""
    encode(point, reference_point, edgelengths) → Vector{Int}

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
encode(rand(3), refpoint, steps)
```
"""
function encode end 

function encode(point::Vector{T}, reference_point, edgelengths) where {T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode(point::SVector{dim, T}, reference_point, edgelengths) where {dim, T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode(point::MVector{dim, T}, reference_point, edgelengths) where {dim, T <: Real}
    floor.(Int, (point .- reference_point) ./ edgelengths)
end

function encode(points::Vector{T}, reference_point, edgelengths) where {T <: Union{Vector, SVector, MVector}}
    [encode(points[i], reference_point, edgelengths) for i = 1:length(points)]
end

function encode(points::AbstractDataset{dim, T}, reference_point, edgelengths) where {dim, T}
    [encode(points[i], reference_point, edgelengths) for i = 1:length(points)]
end