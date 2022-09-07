export entropy_wavelet

include("wavelet_overlap.jl")

"""
    entropy_wavelet(x; wavelet = Wavelets.WT.Daubechies{12}(),
        base = MathConstants.e)

Estimate (Shannon) wave entropy (Rosso, 2001) to the given `base` using maximal overlap
discrete wavelet transform (MODWT).

See also: [`WaveletOverlap`](@ref).

[^Rosso2001]:
    Rosso et al. (2001). Wavelet entropy: a new tool for analysis of short duration
    brain electrical signals. Journal of neuroscience methods, 105(1), 65-75.
"""
function entropy_wavelet(x; wavelet::W = Wavelets.WT.Daubechies{12}(), base = MathConstants.e) where W <:Wavelets.WT.OrthoWaveletClass
    est = WaveletOverlap(wavelet)
    entropy_renyi(x, est; base = base, q = 1)
end
