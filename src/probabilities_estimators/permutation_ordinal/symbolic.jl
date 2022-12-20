
"""
The supertype for probability estimators based on permutation patterns.

Subtypes must implement fields:

- `m::Int` The dimension of the permutation patterns.
- `lt::Function` A function determining how ties are to be broken when constructing
    permutation patterns from embedding vectors.
"""
abstract type PermutationProbabilitiesEstimator <: ProbabilitiesEstimator end

include("common.jl")
include("utils.jl")
include("SymbolicPermutation.jl")
include("SymbolicWeightedPermutation.jl")
include("SymbolicAmplitudeAware.jl")
include("spatial_permutation.jl")
