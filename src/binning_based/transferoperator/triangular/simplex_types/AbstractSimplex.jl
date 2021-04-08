export 
    AbstractSimplex, 
    orientation, 
    intersect,
    radius,
    volume, 
    centroid

import LinearAlgebra: det 
import .Simplices: centroid
import .Simplices: radius 
import .Simplices: orientation
import .Simplices: volume


abstract type AbstractSimplex{D, T} end

# Return the i-th point as column vector
Base.getindex(s::AbstractSimplex, i) = s.vertices[i]
Base.getindex(s::AbstractSimplex, i, j) = s.vertices[i][j]
Base.getindex(s::AbstractSimplex, i::Colon, j) = hcat(s.vertices[j]...,)
Base.getindex(s::AbstractSimplex, i::Colon, j::Colon) = hcat(s.vertices...,)

Base.length(s::AbstractSimplex) = length(s.vertices)
Base.size(s::AbstractSimplex) = length(s[1]), length(s)
Base.size(s::AbstractSimplex, i) = size(s)[i]
Base.IteratorSize(s::AbstractSimplex) = Base.HasLength()

Base.firstindex(s::AbstractSimplex) = 1
Base.lastindex(s::AbstractSimplex) = length(s)
Base.eachindex(s::AbstractSimplex) = Base.OneTo(length(s))
Base.iterate(s::AbstractSimplex, state = 1) = iterate(s.vertices, state)

###########################################################
# Vector{vertex} representation => Array representation
###########################################################
Base.Array(s::AbstractSimplex) = hcat(s.vertices...,)

"""
    dimension(s::AbstractSimplex)

Get the dimension of a simplex.
"""
dimension(s::AbstractSimplex) = length(s[1])
npoints(s::AbstractSimplex) = length(s)

"""
    nvertices(s::AbstractSimplex)

Get the number of vertices of a simplices.
"""
nvertices(s::AbstractSimplex) = length(s)

"""
    orientation(s::AbstractSimplex)

Compute the orientation of a simplex. *Convention: append rows of ones at top of vertex matrix*.
"""
function Simplices.orientation(s::AbstractSimplex)
    det([ones(1, dimension(s) + 1); s[:, :]])
end

"""
    volume(s::AbstractSimplex)

Compute the unscaled volume of the simplex. Divide by factorial(dim) to get the true volume.
"""
Simplices.volume(s::AbstractSimplex) = abs(orientation(s))

function Simplices.centroid(s::AbstractSimplex)
    D = dimension(s)
    # Results in a dim-by-1 matrix, but we just want a vector, so drop the last dimension
    centroid = dropdims(s[:, :] * (ones(D + 1, 1) / (D + 1)), dims = 2)
end

"""
    radius(s::AbstractSimplex)

Compute the radius of a simplex.
"""
function Simplices.radius(s::AbstractSimplex)
    D = dimension(s)
    centroidmatrix = repeat(centroid(s), 1, D + 1)
    radius = sqrt(maximum(ones(1, D) * ((s[:, :] .- centroidmatrix) .^ 2)))
end


function Base.intersect(s1::AbstractSimplex, s2::AbstractSimplex)
    Simplices.simplexintersection(s1[:, :], s2[:, :])
end

"""
    ∩(s1::AbstractSimplex, s2::AbstractSimplex)

Compute the volume intersection between two simplices.
"""
∩(s1::AbstractSimplex, s2::AbstractSimplex) = intersect(s1, s2)

