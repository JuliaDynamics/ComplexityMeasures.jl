export entropy_perm
export entropy_weightedperm
export entropy_ampperm

include("utils.jl")
include("SymbolicPermutation.jl")
include("SymbolicWeightedPermutation.jl")
include("SymbolicAmplitudeAware.jl")
include("spatial_permutation.jl")

"""
    entropy_perm(x; τ = 1, m = 3, base = MathConstants.e)

Compute the (Shannon) permutation entropy (Bandt & Pompe, 2002)[^BandtPompe2002] of order `m` with delay time `τ`.

Short-hand for `entropy_renyi(x, SymbolicPermutation(τ = 1, m = 3), base = base, q = 1)`.

See also: [`SymbolicPermutation`](@ref), [`entropy_renyi`](@ref).

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for time series." Physical review letters 88.17 (2002): 174102.
"""
function entropy_perm(x; τ = 1, m = 3, base = MathConstants.e, lt = isless_rand)
    est = SymbolicPermutation(m = m, τ = τ, lt = lt)
    return entropy_renyi(x, est; base = base, q = 1)
end

"""
    entropy_weightedperm(x; τ = 1, m = 3, base = MathConstants.e)

Compute the (Shannon) weighted permutation entropy (Fadlallah et al., 2013)[^Fadlallah2013] of order `m` with delay time `τ`.

Short-hand for `entropy_renyi(x, SymbolicWeightedPermutation(τ = 1, m = 3), base = base, q = 1)`.

See also: [`SymbolicPermutation`](@ref), [`entropy_renyi`](@ref).

[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical Review E 87.2 (2013): 022911.
"""
function entropy_weightedperm(x; τ = 1, m = 3, base = MathConstants.e, lt = isless_rand)
    est = SymbolicWeightedPermutation(m = m, τ = τ, lt = lt)
    return entropy_renyi(x, est; base = base, q = 1)
end

"""
    entropy_ampperm(x; τ = 1, m = 3, A = 0.5, base = MathConstants.e)

Compiute the (Shannon) amplitude-aware permutation entropy of order `m` with delay time `τ`,
with weighting parameter `A`.

Short-hand for `entropy_renyi(x, SymbolicAmplitudeAwarePermutation(τ = 1, m = 3, A = A), base = base, q = 1)`.

See also: [`SymbolicPermutation`](@ref), [`entropy_renyi`](@ref).

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy:
    Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.
"""
function entropy_ampperm(x; τ = 1, m = 3, A = 0.5, base = MathConstants.e, lt = isless_rand)
    est = SymbolicAmplitudeAwarePermutation(m = m, τ = τ, A = A, lt = lt; base)
    return entropy_renyi(x, est; base = base, q = 1)
end
