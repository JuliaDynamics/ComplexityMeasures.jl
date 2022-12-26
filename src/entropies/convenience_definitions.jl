# This file defines and exports some convenience definition of entropies
# for commonly used names in the literature. They aren't actually new entropies
# as discussed extensively in the documentation.
export entropy_permutation, entropy_wavelet, entropy_dispersion

"""
    entropy_permutation(x; τ = 1, m = 3, base = 2)

Compute the permutation entropy of `x` of order `m` with delay/lag `τ`.
This function is just a convenience call to:

```julia
est = SymbolicPermutation(; m, τ)
entropy(Shannon(base), x, est)
```

See [`SymbolicPermutation`](@ref) for more info. Similarly, one can use
`SymbolicWeightedPermutation` or `SymbolicAmplitudeAwarePermutation`
for the weighted/amplitude-aware versions.
"""
function entropy_permutation(x; base = 2, kwargs...)
    est = SymbolicPermutation(; kwargs...)
    entropy(Shannon(; base), est, x)
end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)

Compute the wavelet entropy. This function is just a convenience call to:

```julia
est = WaveletOverlap(wavelet)
entropy(Shannon(base), est, x)
```

See [`WaveletOverlap`](@ref) for more info.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)
    est = WaveletOverlap(x, wavelet)
    entropy(Shannon(; base), est, x)
end

"""
    entropy_dispersion(x; base = 2, kwargs...)

Compute the dispersion entropy. This function is just a convenience call to:

```julia
est = Dispersion(kwargs...)
entropy(Shannon(base), est, x)
```

See [`Dispersion`](@ref) for more info.
"""
function entropy_dispersion(x; base = 2, kwargs...)
    est = Dispersion(kwargs...)
    entropy(Shannon(; base), est, x)
end
