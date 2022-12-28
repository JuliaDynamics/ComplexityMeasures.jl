export WaveletOverlap
import Wavelets

"""
    WaveletOverlap([wavelet]) <: ProbabilitiesEstimator

Apply the maximal overlap discrete wavelet transform (MODWT) to a
signal, then compute probabilities as the (normalized) energies at different
wavelet scales. These probabilities are used to compute the wavelet entropy,
according to Rosso et al. (2001)[^Rosso2001].
Input timeseries `x` is needed for a well-defined outcome space.

By default the wavelet `Wavelets.WT.Daubechies{12}()`
is used. Otherwise, you may choose a wavelet from the `Wavelets` package
(it must subtype `OrthoWaveletClass`).

## Outcome space

The outcome space for `WaveletOverlap` are the integers `1, 2, …, N` enumerating the
wavelet scales. To obtain a better understanding of what these mean, we
prepared a notebook you can [view online](
https://github.com/kahaaga/waveletentropy_example/blob/main/wavelet_entropy_example.ipynb).
As such, this estimator only works for timeseries input and
input `x` is needed for a well-defined [`outcome_space`](@ref).


[^Rosso2001]:
    Rosso et al. (2001). Wavelet entropy: a new tool for analysis of short duration
    brain electrical signals. Journal of neuroscience methods, 105(1), 65-75.
"""
struct WaveletOverlap{W<:Wavelets.WT.OrthoWaveletClass} <: ProbabilitiesEstimator
    wl::W
end
WaveletOverlap() = WaveletOverlap(Wavelets.WT.Daubechies{12}())

function probabilities_and_outcomes(est::WaveletOverlap, x)
    x isa AbstractVector{<:Real} || error("`WaveletOverlap` only works for timeseries input!")
    p = Probabilities(time_scale_density(x, est.wl))
    return p, 1:length(p)
end

function outcome_space(::WaveletOverlap, x)
    nscales = Wavelets.WT.maxmodwttransformlevels(x)
    return 1:nscales
end

function time_scale_density(x, wl::Wavelets.WT.OrthoWaveletClass)
    W = get_modwt(x, wl)
    return relative_wavelet_energies(W)
end
# maximum overlap discrete wavelet transform
function get_modwt(x, wl)
    orthofilter = Wavelets.wavelet(wl)
    nscales = Wavelets.WT.maxmodwttransformlevels(x)
    return Wavelets.modwt(x, orthofilter, nscales)
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
