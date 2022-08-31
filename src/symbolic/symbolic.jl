export SymbolicProbabilityEstimator

"""
A probability estimator based on symbolization.
"""
abstract type SymbolicProbabilityEstimator <: ProbabilitiesEstimator end

include("interface.jl")
include("utils.jl")
include("SymbolicPermutation.jl")
include("SymbolicWeightedPermutation.jl")
include("SymbolicAmplitudeAware.jl")
include("2d/symbolic_2d.jl")

"""
    permentropy(x; τ = 1, m = 3, base = MathConstants.e)
Shorthand for `genentropy(x, SymbolicPermutation(m = m, τ = τ); base)` which
calculates the permutation entropy of order `m` with delay time `τ` used for embedding.
"""
function permentropy(x; τ = 1, m = 3, base = MathConstants.e)
    return genentropy(x, SymbolicPermutation(m = m, τ = τ); base)
end

export permentropy
