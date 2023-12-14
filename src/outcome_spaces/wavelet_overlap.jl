export WaveletOverlap
import Wavelets

"""
    WaveletOverlap([wavelet]) <: OutcomeSpace

An [`OutcomeSpace`](@ref) based on the maximal overlap discrete wavelet transform (MODWT).

When used with [`probabilities`](@ref), the MODWT is applied to a
signal, then probabilities are computed as the (normalized) energies at different
wavelet scales. These probabilities are used to compute the wavelet entropy
according to [Rosso2001](@citet).
Input timeseries `x` is needed for a well-defined outcome space.

By default the wavelet `Wavelets.WT.Daubechies{12}()`
is used. Otherwise, you may choose a wavelet from the `Wavelets` package
(it must subtype `OrthoWaveletClass`).

## Outcome space

The outcome space for `WaveletOverlap` are the integers `1, 2, …, N` enumerating the
wavelet scales. To obtain a better understanding of what these mean, we
prepared a notebook you can [view online](https://github.com/kahaaga/waveletentropy_example/blob/main/wavelet_entropy_example.ipynb).
As such, this estimator only works for timeseries input and
input `x` is needed for a well-defined [`outcome_space`](@ref).
"""
struct WaveletOverlap{W<:Wavelets.WT.OrthoWaveletClass} <: OutcomeSpace
    wl::W
end
WaveletOverlap() = WaveletOverlap(Wavelets.WT.Daubechies{12}())

function probabilities(o::WaveletOverlap, x)
    if !(x isa AbstractVector{<:Real})
        throw(ArgumentError("`WaveletOverlap` only works for timeseries input!"))
    end
    relative_freqs = time_scale_density(x, o.wl)
    outs = 1:length(relative_freqs)
    return Probabilities(relative_freqs, (x1 = outs,))
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
    tr = Wavelets.modwt(x, orthofilter, nscales)
    return tr[:, 1:end-1] # discard scaling coefficients in last column
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
