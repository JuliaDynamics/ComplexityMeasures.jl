# This file defines and exports some convenience definition of entropies
# for commonly used names in the literature. They aren't actually new entropies
# as discussed extensively in the documentation.

function entropy_permutation(args...)

end

function entropy_spatial_permutation(args...)

end

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = MathConstants.e)

Compute the wavelet entropy. This function is just a convenience call to:
```julia
est = WaveletOverlap(wavelet)
entropy_renyi(x, est; base, q = 1)
```
See [`WaveletOverlap`](@ref) for more.
"""
function entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(), base = MathConstants.e)
    est = WaveletOverlap(wavelet)
    entropy_renyi(x, est; base, q = 1)
end

function entropy_dispersion(args...)

end