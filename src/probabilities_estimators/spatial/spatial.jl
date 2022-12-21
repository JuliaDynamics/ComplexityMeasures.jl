""" A convenience abstract type that makes dispatch for pixel retrieval easier."""
abstract type SpatialProbEst{D, P} <: ProbabilitiesEstimator end

include("utils.jl")
include("spatial_permutation/SpatialSymbolicPermutation.jl")
include("spatial_dispersion/SpatialDispersion.jl")
