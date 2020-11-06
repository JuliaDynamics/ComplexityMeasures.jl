import Entropies.ProbabilitiesEstimator

"""
    WaveletProbabilitiesEstimator <: ProbabilitiesEstimator

Abstract type for probabilities estimators using wavelet transforms. 
These estimators compute probabilities from the energies at different 
wavelet scales.
"""
abstract type WaveletProbabilitiesEstimator <: ProbabilitiesEstimator end


include("TimeScaleMODWT.jl")