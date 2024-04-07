# This file defines and exports some convenience definition of quantities
# for commonly used names in the literature.
export entropy_permutation, entropy_wavelet, entropy_dispersion
export entropy_sample, entropy_approx
export entropy_distribution

"""
    entropy_distribution(x; τ = 1, m = 3, n = 3, base = 2)

Compute the distribution entropy [Li2015](@cite) of `x` using embedding dimension `m`
with delay/lag `τ`, using the Chebyshev distance metric, and using an `n`-element
equally-spaced binning over the distribution of distances to estimate probabilities.

This function is just a convenience call to:

```julia
x = rand(1000000)
o = SequentialPairDistances(x, n, m, τ, metric = Chebyshev())
h = information(Shannon(base = 2), o, x)
```

See [`SequentialPairDistances`](@ref) for more info.
"""
function entropy_distribution(x; base = 2, kwargs...)
    est = SequentialPairDistances(x; kwargs...)
    information(Shannon(; base), est, x)
end

"""
    entropy_permutation(x; τ = 1, m = 3, base = 2)

Compute the permutation entropy of `x` of order `m` with delay/lag `τ`.
This function is just a convenience call to:

```julia
est = OrdinalPatterns(; m, τ)
information(Shannon(base), est, x)
```

See [`OrdinalPatterns`](@ref) for more info. Similarly, one can use
[`WeightedOrdinalPatterns`](@ref) or [`AmplitudeAwareOrdinalPatterns`](@ref)
for the weighted/amplitude-aware versions.
"""
function entropy_permutation(x; base = 2, kwargs...)
    est = OrdinalPatterns(; kwargs...)
    information(Shannon(base), est, x)
end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)

Compute the wavelet entropy. This function is just a convenience call to:

```julia
est = WaveletOverlap(wavelet)
information(Shannon(base), est, x)
```

See [`WaveletOverlap`](@ref) for more info.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = 2)
    est = WaveletOverlap(wavelet)
    information(Shannon(base), est, x)
end

"""
    entropy_dispersion(x; base = 2, kwargs...)

Compute the dispersion entropy. This function is just a convenience call to:

```julia
est = Dispersion(kwargs...)
information(Shannon(base), est, x)
```

See [`Dispersion`](@ref) for more info.
"""
function entropy_dispersion(x; base = 2, kwargs...)
    est = Dispersion(kwargs...)
    information(Shannon(base), est, x)
end

"""
    entropy_sample(x; r = 0.2std(x), m = 2, τ = 1, normalize = true)

Convenience syntax for estimating the (normalized) sample entropy (Richman & Moorman, 2000)
of timeseries `x`.

This is just a wrapper for `complexity(SampleEntropy(; r, m, τ, base), x)`.

See also: [`SampleEntropy`](@ref), [`complexity`](@ref), [`complexity_normalized`](@ref)).
"""
function entropy_sample(x; normalize = true, r = 0.2std(x), kwargs...)
    c = SampleEntropy(x; r, kwargs...)
    if normalize
        complexity_normalized(c, x)
    else
        complexity(c, x)
    end
end

"""
    entropy_approx(x; m = 2, τ = 1, r = 0.2 * Statistics.std(x), base = MathConstants.e)

Convenience syntax for computing the approximate entropy (Pincus, 1991) for timeseries `x`.

This is just a wrapper for `complexity(ApproximateEntropy(; m, τ, r, base), x)` (see
also [`ApproximateEntropy`](@ref)).
"""
function entropy_approx(x; kwargs...)
    c = ApproximateEntropy(x; kwargs...)
    return complexity(c, x)
end
