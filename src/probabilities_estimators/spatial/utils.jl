# This file contains functions that are common to all spatial symbolic estimators.
# --------------------------------------------------------------------------------

# get stencil in the form of vectors of cartesian indices from either input type
stencil_to_offsets(stencil::Vector{CartesianIndex{D}}) where D = stencil, D

function stencil_to_offsets(stencil::NTuple{2, NTuple{D, T}}) where {D, T}
    # get extent and lag from stencil
    extent, lag = stencil
    # generate a D-dimensional stencil
    # start by generating a list of iterators for each dimension
    iters = [0:lag[i]:extent[i]-1 for i in 1:D]
    # then generate the stencil. We use an iterator product that we basically only reshape
    # after that
    stencil = CartesianIndex.(vcat(collect(Iterators.product(iters...))...))
    return stencil, D
end

function stencil_to_offsets(stencil::Array{Int, D}) where D
    # translate D-dim array into stencil of cartesian indices (of dimension D)
    stencil = [idx - CartesianIndex(Tuple(ones(Int, D))) for idx in findall(Bool.(stencil))]
    # subtract first coordinate from everything to get a stencil that contains (0,0)
    stencil = [idx - stencil[1] for idx in stencil]
    return stencil, D
end

"""
    stencil_length(stencil::Vector{CartesianIndex{D}}) where D → length(stencil)
    stencil_length(stencil::Array{Int, D}) where D → sum(stencil)
    stencil_length(stencil::NTuple{2, NTuple{D, T}}) where {D, T} → prod(stencil[1])!

Count the number of elements in the `stencil`.
"""
function stencil_length end
stencil_length(stencil::Vector{CartesianIndex{D}}) where D = length(stencil)
stencil_length(stencil::Array{Int, D}) where D = sum(stencil)
stencil_length(stencil::NTuple{2, NTuple{D, T}}) where {D, T} = factorial(prod(stencil[1]))

function preprocess_spatial(stencil, x::AbstractArray, periodic::Bool = true)
    arraysize = size(x)
    stencil, D = stencil_to_offsets(stencil)
    @assert length(arraysize) == D "Indices and input array must match dimensionality!"
    # Store valid indices for later iteration
    if periodic
        valid = CartesianIndices(x)
    else
        # collect maximum offsets in each dimension for limiting ranges
        maxoffsets = [maximum(s[i] for s in stencil) for i in 1:D]
        # Safety check
        minoffsets = [min(0, minimum(s[i] for s in stencil)) for i in 1:D]
        ranges = Iterators.product(
            [(1-minoffsets[i]):(arraysize[i]-maxoffsets[i]) for i in 1:D]...
        )
        valid = Base.Generator(idxs -> CartesianIndex{D}(idxs), ranges)
    end

    return stencil, arraysize, valid
end

# This source code is a modification of the code of Agents.jl that finds neighbors
# in grid-like spaces. It's the code of `nearby_positions` in `grid_general.jl`.
function pixels_in_stencil(pixel, est::SpatialProbEst{D,false}) where {D}
    @inbounds for i in eachindex(est.stencil)
        est.viewer[i] = est.stencil[i] + pixel
    end
    return est.viewer
end

function pixels_in_stencil(pixel, est::SpatialProbEst{D,true}) where {D}
    @inbounds for i in eachindex(est.stencil)
        # It's annoying that we have to change to tuple and then to CartesianIndex
        # because iteration over cartesian indices is not allowed. But oh well.
        est.viewer[i] = CartesianIndex{D}(
            mod1.(Tuple(est.stencil[i] + pixel), est.arraysize)
        )
    end
    return est.viewer
end
