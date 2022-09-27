# This file defines and exports some convenience definition of entropies
# for commonly used names in the literature. They aren't actually new entropies
# as discussed extensively in the documentation.
export entropy_permutation, entropy_spatial_permutation, entropy_wavelet
export entropy_dispersion, entropy_kernel

"""
    entropy_permutation(x; τ = 1, m = 3, base = 2)

Compute the permutation entropy of order `m` with delay/lag `τ`.
This function is just a convenience call to:
```julia
est = SymbolicPermutation(; m, τ)
entropy(Shannon(base), x, est)
```
See [`SymbolicPermutation`](@ref) for more info.
Similarly, one can use `SymbolicWeightedPermutation` or `SymbolicAmplitudeAwarePermutation`
for the weighted/amplitude-aware versions.
"""
function entropy_permutation(x; τ = 1, m = 3, base = 2)
    est = SymbolicPermutation(; m, τ)
    entropy(Shannon(base), x, est)
end

"""
    entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)

Compute the spatial permutation entropy of `x` given the `stencil`.
Here `x` must be a matrix or higher dimensional `Array` containing spatial data.
This function is just a convenience call to:
```julia
est = SpatialSymbolicPermutation(stencil, x, periodic)
entropy(Renyi(;kwargs...), x, est)
```
See [`SpatialSymbolicPermutation`](@ref) for more info, or how to encode stencils.
"""
function entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)
    est = SpatialSymbolicPermutation(stencil, x, periodic)
    entropy(Renyi(;kwargs...), x, est)
end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)

Compute the wavelet entropy. This function is just a convenience call to:
```julia
est = WaveletOverlap(wavelet)
entropy(Shannon(base), x, est)
```
See [`WaveletOverlap`](@ref) for more info.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)
    est = WaveletOverlap(wavelet)
    entropy(Shannon(base), x, est)
end

"""
    entropy_dispersion(x; m = 2, τ = 1, s = GaussianSymbolization(3),
        base = 2)

Compute the dispersion entropy. This function is just a convenience call to:
```julia
est = Dispersion(m = m, τ = τ, s = s)
entropy(Shannon(base), x, est)
```
See [`Dispersion`](@ref) for more info.
"""
function entropy_dispersion(x; s = GaussianSymbolization(3), base = 2)
    est = Dispersion(m = m, τ = τ, s = s)
    entropy(Shannon(base), x, est)
end

import Statistics
"""
    entropy_kernel(x; ϵ::Real = 0.2*Statistics.std(x), method = KDTree; w = 0,
        metric = Euclidean(), base = 2)

Calculate Shannon entropy using the "naive" kernel density estimation approach (KDE), as
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

This function is just a convenience call to:
```julia
est = NaiveKernel(ϵ, method; w, metric)
entropy(Shannon(base), x, est)
```

[^PrichardTheiler1995]:
    Prichard, D., & Theiler, J. (1995).
    Generalized redundancies for time series analysis.
    Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
function entropy_kernel(x; ϵ::Real = 0.2*Statistics.std(x), method = KDTree, w = 0,
        metric = Euclidean(), base = 2)
    est = NaiveKernel(ϵ, method, w = w, metric = metric)
    return entropy(Shannon(base), x, est)
end
