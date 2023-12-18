export PowerSpectrum
import FFTW

"""
    PowerSpectrum() <: OutcomeSpace

An [`OutcomeSpace`](@ref) based on the power spectrum of a timeseries (amplitude square of
its Fourier transform).

If used with [`probabilities`](@ref), then the spectrum normalized to sum = 1
is returned as probabilities.
The Shannon entropy of these probabilities is typically referred in the literature as
_spectral entropy_, e.g. [Llanos2017](@citet) and [Tian2017](@citet).

The closer the spectrum is to flat, i.e., white noise, the higher the entropy. However,
you can't compare entropies of timeseries with different length, because the binning
in spectral space depends on the length of the input.

## Outcome space

The outcome space `Ω` for `PowerSpectrum` is the set of frequencies in Fourier space. They
should be multiplied with the sampling rate of the signal, which is assumed to be `1`.
Input `x` is needed for a well-defined [`outcome_space`](@ref).
"""
struct PowerSpectrum <: OutcomeSpace end

function probabilities_and_outcomes(::PowerSpectrum, x)
    if !(x isa AbstractVector{<:Real})
        throw(ArgumentError("`PowerSpectrum` only works for timeseries input!"))
    end
    f = FFTW.rfft(x)
    probs = Probabilities(abs2.(f))
    outs = FFTW.rfftfreq(length(x))
    p = Probabilities(probs, outs)
    return p, outcomes(p)
end

outcome_space(::PowerSpectrum, x) = FFTW.rfftfreq(length(x))

function total_outcomes(::PowerSpectrum, x)
    n = length(x)
    # From the docstring of `AbstractFFTs.rfftfreq`:
    iseven(n) ? length(0:(n÷2)) : length(0:((n-1)÷2))
end
