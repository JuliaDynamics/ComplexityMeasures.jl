"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilitiesEstimator <: ProbabilitiesEstimator end

function outcomes(x::AbstractVector{T}, est::PermutationProbabilitiesEstimator) where T<:Real
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    @show genembed(x, τs).data
    return outcomes(genembed(x, τs), OrdinalPatternEncoding(m = est.m, lt = est.lt))
end

include("utils.jl")
include("SymbolicPermutation.jl")
include("SymbolicWeightedPermutation.jl")
include("SymbolicAmplitudeAware.jl")
include("spatial_permutation.jl")
