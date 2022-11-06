""" A convenience abstract type that makes dispatch for pixel retrieval easier."""
abstract type SpatialProbEst{D, P} <: ProbabilitiesEstimator end

include("utils.jl")
include("permutation_ordinal/SpatialSymbolicPermutation.jl")
include("dispersion/SpatialDispersion.jl")
