""" A convenience abstract type that makes dispatch for pixel retrieval easier."""
abstract type SpatialProbEst{D, P} <: CountBasedOutcomeSpace end

include("utils.jl")
include("spatial_ordinal/SpatialOrdinalPatterns.jl")
include("spatial_dispersion/SpatialDispersion.jl")
