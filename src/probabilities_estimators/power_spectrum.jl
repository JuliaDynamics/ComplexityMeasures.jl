export PowerSpectrum
import FFTW

"""
    PowerSpectrum(x_or_length(x)) <: ProbabilitiesEstimator

Calculate the power spectrum of a timeseries (amplitude square of its Fourier transform),
and return the spectrum normalized to sum = 1 as probabilities.
The Shannon entropy of these probabilities is typically referred in the literature as
_spectral entropy_, e.g. [^Llanos2016],[^Tian2017].

The closer the spectrum is to flat, i.e., white noise, the higher the entropy. However,
you can't compare entropies of timeseries with different length, because the binning
in spectral space depends on the length of the input.

## Outcome space

The outcome space `Ω` for `PowerSpectrum` is the set of frequencies in Fourier space. They
should be multiplied with the sampling rate of the signal, which is assumed to be `1`.
The length of the input is therefore required for this estimator to have a well-defined
outcome space.

[^Llanos2016]:
    Llanos et al., _Power spectral entropy as an information-theoretic correlate of manner
    of articulation in American English_, [The Journal of the Acoustical Society of America
    141, EL127 (2017)](https://doi.org/10.1121/1.4976109)

[^Tian2017]:
    Tian et al, _Spectral EntropyDefinition Can Predict Changes of Working Memory Performance Reduced
    by Short-Time Training in the Delayed-Match-to-Sample Task_,
    [Front. Hum. Neurosci.](https://doi.org/10.3389/fnhum.2017.00437)
"""
struct PowerSpectrum <: ProbabilitiesEstimator
    length::Int
end
PowerSpectrum(x::AbstractVector) = PowerSpectrum(length(x))

function probabilities_and_outcomes(est::PowerSpectrum, x)
    probs = probabilities(est, x)
    events = FFTW.rfftfreq(est.length)
    return probs, events
end

outcome_space(est::PowerSpectrum) = FFTW.rfftfreq(est.length)

function probabilities(::PowerSpectrum, x)
    @assert x isa AbstractVector{<:Real} "`PowerSpectrum` only works for timeseries input!"
    f = FFTW.rfft(x)
    return Probabilities(abs2.(f))
end

function total_outcomes(est::PowerSpectrum)
    n = est.length
    # From the docstring of `AbstractFFTs.rfftfreq`:
    iseven(n) ? length(0:(n÷2)) : length(0:((n-1)÷2))
end
