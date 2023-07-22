# from before histogram-from-ranges rework
function FixedRectangularBinning(ϵmin::NTuple{D,T}, ϵmax::NTuple{D,T}, N::Int) where {D, T}
    FixedRectangularBinning(ntuple(i->range(ϵmin[i],ϵmax[i];length=N), D))
end
function FixedRectangularBinning(ϵmin::Real, ϵmax::Real, N, D::Int = 1)
    if N isa Int
        FixedRectangularBinning(ntuple(x-> range(ϵmin, nextfloat(float(ϵmax)); length = N), D))
    else
        FixedRectangularBinning(ntuple(x-> range(ϵmin, nextfloat(float(ϵmax)); step = N), D))
    end
end


# from before https://github.com/JuliaDynamics/ComplexityMeasures.jl/pull/239
function entropy(e::InformationMeasure, est::DifferentialInfoEstimator, x)
    if e isa Shannon
        return information(est, x)
    else
        throw(ErrorException("only shannon entropy supports this deprecated interface"))
    end
end

@deprecate ComplexityMeasure ComplexityEstimator
@deprecate EntropyDefinition InformationMeasure
@deprecate entropy_maximum information_maximum
@deprecate DifferentialEntropyEstimator DifferentialInfoEstimator
@deprecate DiscreteEntropyEstimator DiscreteInfoEstimator
@deprecate MLEntropy PlugIn

# From before 2.0:
@deprecate TimeScaleMODWT WaveletOverlap
function probabilities(x::Vector_or_SSSet, ε::Union{Real, Vector{<:Real}})
    @warn """
    `probabilities(x::Vector_or_SSSet, ε::Real)`
    is deprecated, use `probabilities(ValueHistogram(ε), x)`.
    """
    probabilities(ValueHistogram(ε), x)
end

function probabilities(x, est::ProbabilitiesEstimator)
    @warn """
    `probabilities(x, est::ProbabilitiesEstimator)`
    is deprecated, use `probabilities(est::ProbabilitiesEstimator, x) instead`.
    """
    return probabilities(est, x)
end

export genentropy, permentropy

function permentropy(x; τ = 1, m = 3, base = MathConstants.e)
    @warn """
    `permentropy(x; τ, m, base)` is deprecated.
    Use instead: `entropy_permutation(x; τ, m, base)`, or even better, use the
    direct syntax discussed in the docstring of `entropy_permutation`.
    """
    return entropy_permutation(x; τ, m, base)
end

function genentropy(probs::Probabilities; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(probs::Probabilities; q, base)` deprecated.
    Use instead: `information(Renyi(q, base), probs)`.
    """
    return information(Renyi(q, base), probs)
end

function genentropy(x::Array_or_SSSet, ε::Real; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_SSSet, ε::Real; q, base)` is deprecated.
    Use instead: `information(Renyi(q, base), ValueHistogram(ε), x)`.
    """
    return information(Renyi(q, base), ValueHistogram(ε), x)
end

function genentropy(x::Array_or_SSSet, est::ProbabilitiesEstimator; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_SSSet, est::ProbabilitiesEstimator; q, base)` is deprecated.
    Use instead: `information(Renyi(q, base), est, x)`.
    """
    return information(Renyi(q, base), est, x)
end

####################################################################################
# For 3.0
####################################################################################
export entropy
entropy(args...) = information(args...)


# Discrete
################################################################################
function entropy(e::InformationMeasure, est::ProbabilitiesEstimator, x)
    @warn """
    `entropy(e::EntropyDefinition, est::ProbabilitiesEstimator, x)` is deprecated.
    From 3.0 onwards, use `information(PlugIn(measure = e), est, x)` instead.
    """
    return information(PlugIn(e), est, x)
end

export entropy_normalized
entropy_normalized(args...) = information_normalized(args...)

function entropy_normalized(e::InformationMeasure, est::ProbabilitiesEstimator, x)
    @warn """
    `entropy_normalized(e::EntropyDefinition, est::ProbabilitiesEstimator, x)` is deprecated.
    From 3.0 onwards, use `information_normalized(PlugIn(measure = e), est, x)` instead.
    """
    return information_normalized(PlugIn(e), est, x)
end

# Differential
################################################################################
function entropy(est::DifferentialInfoEstimator, x)
    @warn """
    `entropy(est::DifferentialEntropyEstimator, x)` is deprecated.
    Use `information(est, x)` instead.
    """
    return information(est, x)
end
