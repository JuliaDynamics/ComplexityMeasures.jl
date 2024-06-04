""" A convenience abstract type that makes dispatch for pixel retrieval easier."""
abstract type SpatialOutcomeSpace{D, P} <: CountBasedOutcomeSpace end

include("utils.jl")
include("spatial_ordinal/SpatialOrdinalPatterns.jl")
include("spatial_dispersion/SpatialDispersion.jl")
include("spatial_bubble_sort_swaps/spatial_bubble_sort_swaps.jl")