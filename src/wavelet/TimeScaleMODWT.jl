export TimeScaleMODWT
import Wavelets
import Wavelets: wavelet, maxdyadiclevel, modwt

"""
    TimeScaleMODWT([wavelet]) <: WaveletProbabilitiesEstimator

Apply the maximal overlap discrete wavelet transform (MODWT) to a
signal, then compute probabilities/entropy from the energies at different
wavelet scales. This implementation is based on Rosso et
al. (2001)[^Rosso2001].
Optionally specify a wavelet to be used.

The probability `p[i]` is the relative/total energy for the `i`-th wavelet scale.
To obtain a better understand of what these probabilities mean, we prepared
a notebook you can [view online](
https://github.com/kahaaga/waveletentropy_example/blob/main/wavelet_entropy_example.ipynb)

By default the wavelet `Wavelets.WT.Daubechies{12}()`
is used. Otherwise, you may choose a wavelet from the `Wavelets` package
(it myst subtype `OrthoWaveletClass`).

[^Rosso2001]:
    Rosso et al. (2001). Wavelet entropy: a new tool for analysis of short duration
    brain electrical signals. Journal of neuroscience methods, 105(1), 65-75.
"""
struct TimeScaleMODWT <: WaveletProbabilitiesEstimator
    wl::Wavelets.WT.OrthoWaveletClass
end
TimeScaleMODWT() = TimeScaleMODWT(Wavelets.WT.Daubechies{12}())

function probabilities(x, est::TimeScaleMODWT)
    @assert x isa AbstractVector{<:Real} "`TimeScaleMODWT` only works for timeseries input!"
    Probabilities(time_scale_density(x, est.wl))
end

function time_scale_density(x, wl::Wavelets.WT.OrthoWaveletClass)
    W = get_modwt(x, wl)
    return relative_wavelet_energies(W)
end
# maximum overlap discrete wavelet transform
function get_modwt(x, wl)
    orthofilter = wavelet(wl)
    nscales = maxdyadiclevel(x)
    return WaveletsW.modwt(x, orthofilter, nscales)
end

function relative_wavelet_energies(W::AbstractMatrix)
    js = 1:size(W, 2)
    if any(j ∉ 1:size(W, 2) for j in js)
        error("scales $(js) contains scales not present in wavelet coefficient "*
              "matrix with scales j ∈ 1:$(size(W, 2))")
    end
    total_energy = sum(W .^ 2)
    return [energy_at_scale(W, j) / total_energy for j in js]
end

energy_at_scale(W, j::Int) = sum(w*w for w in @view(W[:, j]))
energy_at_time(W, t::Int) = sum(w*w for w in @view(W[t, :]))
