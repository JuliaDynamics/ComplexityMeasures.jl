export TimeScaleMODWT
import Wavelets
import Wavelets: wavelet, maxdyadiclevel, modwt

"""
    TimeScaleMODWT <: WaveletProbabilitiesEstimator
    TimeScaleMODWT(wl::Wavelets.WT.OrthoWaveletClass = Wavelets.WT.Daubechies{12}())

Apply the maximal overlap discrete wavelet transform (MODWT) to a
signal, then compute probabilities/entropy from the energies at different
wavelet scales. This implementation is based on Rosso et
al. (2001)[^Rosso2001].
Optionally specify a wavelet to be used.

The probability `p[i]` is the relative/total energy for the i-th wavelet scale.

## Example

Manually picking a wavelet is done as follows.

```julia
using Entropies, Wavelets
N = 200
a = 10
t = LinRange(0, 2*a*π, N)
x = sin.(t .+  cos.(t/0.1)) .- 0.1;

# Pick a wavelet (if no wavelet provided, defaults to Wavelets.WL.Daubechies{12}())
wl = Wavelets.WT.Daubechies{12}()

# Compute the probabilities (relative energies) at the different wavelet scales
probabilities(x, TimeScaleMODWT(wl))
```

If no wavelet provided, the default is `Wavelets.WL.Daubechies{12}())`.

[^Rosso2001]: Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A., Schürmann, M., & Başar, E. (2001). Wavelet entropy: a new tool for analysis of short duration brain electrical signals. Journal of neuroscience methods, 105(1), 65-75.
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
    ps = relative_wavelet_energies(W)
end
# maximum overlap discrete wavelet transform
function get_modwt(x, wl)
    orthofilter = wavelet(wl)
    nscales = maxdyadiclevel(x)
    W = modwt(x, orthofilter, nscales)
end

function relative_wavelet_energies(W::AbstractMatrix, js = 1:size(W, 2))
    if any(j ∉ 1:size(W, 2) for j in js)
        error("scales $(js) contains scales not present in wavelet coefficient "*
              "matrix with scales j ∈ 1:$(size(W, 2))")
    end
    total_energy = sum(W .^ 2)
    return [energy_at_scale(W, j) / total_energy for j in js]
end

function energy_at_scale(W::AbstractArray{T, 2}, j::Int) where T<:Real
    1 <= j <= size(W, 2) || error("Scale j does not exist in wave coefficient matrix W. Available scales are j=1:$(size(W, 2))")
    Eⱼ = sum(w*w for w in @view(W[:, j]))
end

# This function is not used anywhere currently
function energy_at_time(W::AbstractArray{T, 2}, t::Int) where T<:Real
    1 <= t <= size(W, 1) || error("Time t does not exist in wave coefficient matrix W. Available times are t=1:$(size(W, 1))")
    Eₜ = sum(w*w for w in @view(W[t, :]))
end
