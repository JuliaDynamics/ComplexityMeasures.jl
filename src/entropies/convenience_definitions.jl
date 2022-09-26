# This file defines and exports some convenience definition of entropies
# for commonly used names in the literature. They aren't actually new entropies
# as discussed extensively in the documentation.
export entropy_shannon
export entropy_permutation, entropy_spatial_permutation, entropy_wavelet

export entropy_shannon
export maxentropy_shannon

"""
    entropy_shannon(args...; base = 2)

Equivalent to `entropy(Renyi(;base), args...)` and provided solely for convenience.
Computes the Shannon entropy, given by
```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
"""
entropy_shannon(args...; base = 2) = entropy(Renyi(; base), args...)

"""
    entropy_permutation(x; τ = 1, m = 3, base = 2)

Compute the permutation entropy of order `m` with delay/lag `τ`.
This function is just a convenience call to:
```julia
est = SymbolicPermutation(; m, τ)
entropy_shannon(x, est; base)
```
See [`SymbolicPermutation`](@ref) for more info.
Similarly, one can use `SymbolicWeightedPermutation` or `SymbolicAmplitudeAwarePermutation`
for the weighted/amplitude-aware versions.
"""
function entropy_permutation(x; τ = 1, m = 3, base = 2)
    est = SymbolicPermutation(; m, τ)
    entropy_shannon(x, est; base)
end

"""
    entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)

Compute the spatial permutation entropy of `x` given the `stencil`.
Here `x` must be a matrix or higher dimensional `Array` containing spatial data.
This function is just a convenience call to:
```julia
est = SpatialSymbolicPermutation(stencil, x, periodic)
entropy_shannon(x, est; kwargs...)
```
See [`SpatialSymbolicPermutation`](@ref) for more info, or how to encode stencils.
"""
function entropy_spatial_permutation(x, stencil, periodic = true; kwargs...)
    est = SpatialSymbolicPermutation(stencil, x, periodic)
    entropy_shannon(x, est; kwargs...)
end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)

Compute the wavelet entropy. This function is just a convenience call to:
```julia
est = WaveletOverlap(wavelet)
entropy_shannon(x, est; base)
```
See [`WaveletOverlap`](@ref) for more info.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)
    est = WaveletOverlap(wavelet)
    entropy_renyi(x, est; base, q = 1)
end

"""
    entropy_dispersion(x; m = 2, τ = 1, s = GaussianSymbolization(3),
        base = 2)

Compute the dispersion entropy. This function is just a convenience call to:
```julia
est = Dispersion(m = m, τ = τ, s = s)
entropy_shannon(x, est; base)
```
See [`Dispersion`](@ref) for more info.
"""
function entropy_dispersion(x; s = GaussianSymbolization(3),
        base = 2)

    est = Dispersion(m = m, τ = τ, s = s)
    entropy_renyi(x, est; base, q = 1)
end

import Statistics
"""
entropy_kernel(x; ϵ::Real = 0.2*Statistics.std(x), method = KDTree; w = 0,
    metric = Euclidean(), base = 2)

Calculate Shannon entropy using the "naive" kernel density estimation approach (KDE), as
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Shorthand for `entropy(Renyi(;base), x, NaiveKernel(ϵ, method; w, metric))`.


[^PrichardTheiler1995]:
    Prichard, D., & Theiler, J. (1995).
    Generalized redundancies for time series analysis.
    Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
function entropy_kernel(x; ϵ::Real = 0.2*Statistics.std(x), method = KDTree, w = 0,
        metric = Euclidean(), base = 2)
    est = NaiveKernel(ϵ, method, w = w, metric = metric)
    return entropy_renyi(x, est; base = base, q = 1)
end
