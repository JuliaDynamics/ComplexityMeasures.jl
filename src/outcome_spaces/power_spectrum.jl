export PowerSpectrum
import FFTW

"""
    PowerSpectrum() <: OutcomeSpaceModel

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
Input `x` is needed for a well-defined [`outcome_space`](@ref).

[^Llanos2016]:
    Llanos et al., _Power spectral entropy as an information-theoretic correlate of manner
    of articulation in American English_, [The Journal of the Acoustical Society of America
    141, EL127 (2017)](https://doi.org/10.1121/1.4976109)

[^Tian2017]:
    Tian et al, _Spectral InformationMeasure Can Predict Changes of Working Memory Performance Reduced
    by Short-Time Training in the Delayed-Match-to-Sample Task_,
    [Front. Hum. Neurosci.](https://doi.org/10.3389/fnhum.2017.00437)
"""
struct PowerSpectrum <: OutcomeSpaceModel end

function frequencies_and_outcomes(est::PowerSpectrum, x)
    @assert x isa AbstractVector{<:Real} "`PowerSpectrum` only works for timeseries input!"
    f = FFTW.rfft(x)
    probs = Probabilities(abs2.(f))
    events = FFTW.rfftfreq(length(x))
    # Convert to pseudo-counts
    freqs = floor.(Int, probs .* 10e8)

    return freqs, events
end

outcome_space(::PowerSpectrum, x) = FFTW.rfftfreq(length(x))

function total_outcomes(::PowerSpectrum, x)
    n = length(x)
    # From the docstring of `AbstractFFTs.rfftfreq`:
    iseven(n) ? length(0:(n÷2)) : length(0:((n-1)÷2))
end
