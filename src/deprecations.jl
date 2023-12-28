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

@deprecate ComplexityMeasure ComplexityEstimator
@deprecate EntropyDefinition InformationMeasure
@deprecate entropy_maximum information_maximum
@deprecate entropy_normalized information_normalized
@deprecate DifferentialEntropyEstimator DifferentialInfoEstimator
@deprecate DiscreteEntropyEstimator DiscreteInfoEstimator
@deprecate MLEntropy PlugIn

"""
    VisitationFrequency

An alias for [`ValueBinning`](@ref).
"""
const VisitationFrequency = ValueBinning

"""
    ValueHistogram

An alias for [`ValueBinning`](@ref).
"""
const ValueHistogram = ValueBinning

export ValueHistogram, VisitationFrequency

@deprecate CountOccurrences UniqueElements

# From before 2.0:
@deprecate TimeScaleMODWT WaveletOverlap
function probabilities(x::Vector_or_SSSet, ε::Union{Real, Vector{<:Real}})
    @warn """
    `probabilities(x::Vector_or_SSSet, ε::Real)`
    is deprecated, use `probabilities(ValueBinning(ε), x)`.
    """
    probabilities(ValueBinning(ε), x)
end

function probabilities(x, est::OutcomeSpace)
    @warn """
    `probabilities(x, est::OutcomeSpace)`
    is deprecated, use `probabilities(est::OutcomeSpace, x) instead`.
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
    Use instead: `information(Renyi(q, base), ValueBinning(ε), x)`.
    """
    return information(Renyi(q, base), RelativeAmount(), ValueBinning(ε), x)
end

function genentropy(x::Array_or_SSSet, o::OutcomeSpace; q = 1.0, base = MathConstants.e)
    @warn """
    `genentropy(x::Array_or_SSSet, est::ProbabilitiesEstimator; q, base)` is deprecated.
    Use instead: `information(Renyi(q, base), est, x)`.
    """
    return information(Renyi(q, base), RelativeAmount(), o, x)
end

@deprecate SymbolicPermutation OrdinalPatterns
@deprecate SymbolicWeightedPermutation WeightedOrdinalPatterns
@deprecate SymbolicAmplitudeAwarePermutation AmplitudeAwareOrdinalPatterns

function OrdinalPatternEncoding(m::Int, lt::F = isless_rand) where {F}
    @warn "Passing `m` as an argument to `OrdinalPattern...(m = ...)` is deprecated. "*
    "Pass it as a type parameter instead: `OrdinalPattern...{m}`."
    return OrdinalPatternEncoding{m, F}(zero(MVector{m, Int}), lt)
end
# Initializations
function OrdinalPatterns(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return OrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ)
end
function WeightedOrdinalPatterns(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return WeightedOrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ)
end
function AmplitudeAwareOrdinalPatterns(; A = 0.5, τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return AmplitudeAwareOrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ, A)
end

# For 3.0
export allprobabilities
function allprobabilities(args...)
    @warn "`allprobabilities` is deprecated. Use `allprobabilities_and_outcomes` instead."
    return first(allprobabilities_and_outcomes(args...))
end

export allcounts
function allcounts(args...)
    @warn "`allcounts` is deprecated. Use `allcounts_and_outcomes` instead."
    return first(allcounts_and_outcomes(args...))
end