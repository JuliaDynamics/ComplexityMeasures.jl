export TimeScaleMODWT, genentropy, probabilities
import Wavelets
import Wavelets: wavelet, maxdyadiclevel, modwt

"""
    TimeScaleMODWT <: WaveletProbabilitiesEstimator

Apply the maximal overlap discrete wavelet transform (MODWT) to a 
signal, then compute probabilities/entropy from the energies at different 
wavelet scales. This implementation is based on Rosso et 
al. (2001)[^Rosso2001].

    TimeScaleMODWT(wl::Wavelets.WT.OrthoWaveletClass = Wavelets.WT.Daubechies{12}())

Construct a TimeScaleMODWT probabilities/entropy estimator with wavelet `wl`.

## Example 

Manually picking a wavelet is done as follows. 

```jldoctest
using Entropies, Wavelets
wl = Wavelets.WT.Daubechies{4}()
est = TimeScaleMODWT(wl)

# output

TimeScaleMODWT(Wavelets.WT.Daubechies{4}())
```

If no wavelet provided, the default is `Wavelets.WL.Daubechies{12}())`. 

```jldoctest
using Entropies, Wavelets
est = TimeScaleMODWT()

# output

TimeScaleMODWT(Wavelets.WT.Daubechies{12}())
```

[^Rosso2001]: Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A., Schürmann, M., & Başar, E. (2001). Wavelet entropy: a new tool for analysis of short duration brain electrical signals. Journal of neuroscience methods, 105(1), 65-75.
"""
struct TimeScaleMODWT <: WaveletProbabilitiesEstimator
    wl::Wavelets.WT.OrthoWaveletClass
    function TimeScaleMODWT(wl::Wavelets.WT.OrthoWaveletClass = Wavelets.WT.Daubechies{12}())
        new(wl)
    end
end

function get_modwt(x::AbstractVector{T}, wl::Wavelets.WT.OrthoWaveletClass = Wavelets.WT.Daubechies{12}()) where T<:Real
    orthofilter = wavelet(wl)
    nscales = maxdyadiclevel(x)
    W = modwt(x, orthofilter, nscales)
end

function energy_at_scale(W::AbstractArray{T, 2}, j::Int) where T<:Real
    1 <= j <= size(W, 2) || error("Scale j does not exist in wave coefficient matrix W. Available scales are j=1:$(size(W, 2))")
    Eⱼ = sum(W[:, j] .^ 2)
end

function energy_at_time(W::AbstractArray{T, 2}, t::Int) where T<:Real
    1 <= t <= size(W, 1) || error("Time t does not exist in wave coefficient matrix W. Available times are t=1:$(size(W, 1))")
    Eⱼ = sum(W[t, :] .^ 2)
end

function energy_total(W::AbstractArray{T, 2}) where T<:Real
    Etot = sum(W .^ 2)
end

function relative_wavelet_energy(W::AbstractArray{T, 2}, j::Int) where T<:Real
    energy_at_scale(W, j) / energy_total(W)
end

function relative_wavelet_energies(W::AbstractArray{T, 2}, js = 1:size(W, 2)) where T<:Real
    all(1 .<= js .<= size(W, 2)) || error(ArgumentError("scales $(js) contains scales not present in wavelet coefficient matrix with scales j=1:$(size(W, 2))"))
    [energy_at_scale(W, j) / energy_total(W) for j in js]
end

function time_scale_density(x::AbstractVector{T}, wl::Wavelets.WT.OrthoWaveletClass = WT.Daubechies{12}()) where T
    W = get_modwt(x, wl)
    Pⱼs = relative_wavelet_energies(W)
end

"""
# Wavelet-based time-scale probability estimation 

    probabilities(x::AbstractVector{<:Real}, est::TimeScaleMODWT) → ps::Probabilities

Compute the probability distribution of energies from a maximal overlap discrete wavelet 
transform (MODWT) of `x`. The probability `ps[i]` is the relative/total energy for the 
i-th wavelet scale.

```julia
using Entropies, Wavelets
N = 200
a = 10
t = LinRange(0, 2*a*π, N)
x = sin.(t .+  cos.(t/0.1)) .- 0.1;

# Pick a wavelet (if no wavelet provided, defaults to Wavelets.WL.Daubechies{12}())
wl = Wavelets.WT.Daubechies{12}()

# Compute the probabilities (relative energies) at the different wavelet scales
Entropies.probabilities(x, TimeScaleMODWT(wl))
```

See also: [`TimeScaleMODWT`](@ref).
"""
function probabilities(x::AbstractVector{T}, est::TimeScaleMODWT) where T<:Real
    Probabilities(time_scale_density(x, est.wl))
end

"""
# Wavelet-based time-scale entropy

    genentropy(x::AbstractVector{<:Real}, est::TimeScaleMODWT; α = 1, base = 2) → h::Real

Compute the generalized order-`α` time-scale entropy of `x`, from a maximal overlap 
discrete wavelet transform (MODWT) of `x`.

## Example 

```julia
using Entropies, Wavelets
N = 200
a = 10
t = LinRange(0, 2*a*π, N)
x = sin.(t .+  cos.(t/0.1)) .- 0.1;

# Pick a wavelet (if no wavelet provided, defaults to Wavelets.WL.Daubechies{12}())
wl = Wavelets.WT.Daubechies{12}()

# Compute generalized (time-scale) entropy of order 1
Entropies.genentropy(x, TimeScaleMODWT(wl))
```

See also: [`TimeScaleMODWT`](@ref).
"""
genentropy(x::AbstractVector{T}, est::WaveletProbabilitiesEstimator; α::Real) where T<:Real

function genentropy(x::AbstractVector{T}, est::TimeScaleMODWT; α::Real = 1, base = 2) where T<:Real
    ps = probabilities(x, est)
    Entropies.genentropy(ps, α = α, base = base)
end