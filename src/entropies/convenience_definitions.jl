# This file defines and exports some convenience definition of entropies
# for commonly used names in the literature. They aren't actually new entropies
# as discussed extensively in the documentation.
export entropy_permutation, entropy_spatial_permutation, entropy_wavelet

"""
    entropy_permutation(x; τ = 1, m = 3, base = MathConstants.e)

Compute the permutation entropy of order `m` with delay/lag `τ`.
This function is just a convenience call to:
```julia
est = SymbolicPermutation(; m, τ)
entropy_renyi(x, est; base, q = 1)
```
See [`SymbolicPermutation`](@ref) for more info.
Similarly, one can use `SymbolicWeightedPermutation` or `SymbolicAmplitudeAwarePermutation`
for the weighted/amplitude-aware versions.
"""
function entropy_permutation(x; τ = 1, m = 3, base = MathConstants.e)
    est = SymbolicPermutation(; m, τ)
    entropy_renyi(x, est; base, q = 1)
end

"""
    entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)

Compute the spatial permutation entropy of `x` given the `stencil`.
Here `x` must be a matrix or higher dimensional `Array` containing spatial data.
This function is just a convenience call to:
```julia
est = SpatialSymbolicPermutation(stencil, x, periodic)
entropy_renyi(x, est; kwargs..., q = 1)
```
See [`SpatialSymbolicPermutation`](@ref) for more info, or how to encode stencils.
"""
function entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)
    est = SpatialSymbolicPermutation(stencil, x, periodic)
    entropy_renyi(x, est; kwargs..., q = 1)
end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = MathConstants.e)

Compute the wavelet entropy. This function is just a convenience call to:
```julia
est = WaveletOverlap(wavelet)
entropy_renyi(x, est; base, q = 1)
```
See [`WaveletOverlap`](@ref) for more info.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = MathConstants.e)
    est = WaveletOverlap(wavelet)
    entropy_renyi(x, est; base, q = 1)
end

function entropy_dispersion(args...)

end