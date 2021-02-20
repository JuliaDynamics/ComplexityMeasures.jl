
import Distributions: Uniform
import Statistics: std
import StaticArrays: MArray, SArray, MVector, SVector

"""
    addnoise!(pts, joggle_factor)

Adding uniformly distributed noise to each observation of maximum magnitude `joggle_factor`.
"""
function addnoise!(pts::AbstractArray{T, 2}; joggle_factor = 0.001) where T

    D = minimum(size(pts))
    npts = maximum(size(pts))

    if size(pts, 1) > size(pts, 2)
        dim = 1
    else
        dim = 2
    end

    # Scale standard deviation along each axis by joggle factor
    σ = joggle_factor .* std(pts, dims = dim)

    for i in 1:D
        r = [rand(Uniform(-σ[i], σ[i])) for pt in 1:npts]
        if dim == 1
            pts[:, i] .+= r
        elseif dim == 2
            pts[i, :] .+= r
        end
    end
end

"""
    addnoise!(pts, joggle_factor)

Adding uniformly distributed noise to each observation of maximum magnitude `joggle_factor`.
"""
function addnoise!(pts::Vector{Vector{T}}; joggle_factor = 1e-8) where T

    D = length(pts[1])
    npts = length(pts)

    # Scale standard deviation along each axis by joggle factor
    σ = joggle_factor

    for i in 1:D
        r = rand(Uniform(-σ, σ))
        pts[i] .+= r
    end
end

import DelayEmbeddings:
	Dataset

function addnoise!(pts::Dataset; joggle_factor = 1e-8)

    D = size(pts, 2)
    npts = length(pts)

    # Scale standard deviation along each axis by joggle factor
    σ = joggle_factor

    Dataset([pts[i] .+ rand(Uniform(-σ, σ)) for i = 1:npts])
end




"""
    addnoise!(pts, joggle_factor)

Adding uniformly distributed noise to each observation of maximum magnitude `joggle_factor`.
"""
function addnoise!(pts::Vector{MVector{D, T}}; joggle_factor = 1e-8) where {D, T}

    npts = length(pts)

    # Scale standard deviation along each axis by joggle factor
    σ = joggle_factor

    for i in 1:D
        r = rand(Uniform(-σ, σ))
        pts[i] .+= r
    end
end




"""
    addnoise!(pts, joggle_factor)

Adding uniformly distributed noise to each observation of maximum magnitude `joggle_factor`.
"""
function addnoise!(pts::Vector{SVector{D, T}}; joggle_factor = 1e-8) where {D, T}

    npts = length(pts)

    # Scale standard deviation along each axis by joggle factor
    σ = joggle_factor

    [pts[i] .+ rand(Uniform(-σ, σ)) for i = 1:npts]
end
