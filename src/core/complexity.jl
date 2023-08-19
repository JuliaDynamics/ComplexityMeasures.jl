export ComplexityEstimator
export complexity
export complexity_normalized

"""
    ComplexityEstimator

Supertype for estimators for various complexity measures that are not entropies
in the strict mathematical sense.

See [`complexity`](@ref) for all available estimators.
"""
abstract type ComplexityEstimator end

"""
    complexity(c::ComplexityEstimator, x) → m::Real

Estimate a complexity measure according to `c`
for [input data](@ref input_data) `x`, where `c` is an instance of
any subtype of [`ComplexityEstimator`](@ref):

- [`ApproximateEntropy`](@ref).
- [`LempelZiv76`](@ref).
- [`MissingDispersionPatterns`](@ref).
- [`ReverseDispersion`](@ref).
- [`SampleEntropy`](@ref).
- [`StatisticalComplexity`](@ref).
"""
function complexity(c::ComplexityEstimator, x)
    T = typeof(x)
    msg = "`complexity` not implemented for $(typeof(c)) and input data of type $T."
    throw(ArgumentError(msg))
end

"""
    complexity_normalized(c::ComplexityEstimator, x) → m::Real ∈ [a, b]

The same as [`complexity`](@ref), but the result is normalized to the interval `[a, b]`,
where `[a, b]` depends on `c`.
"""
function complexity_normalized(c::C, x) where {C <: ComplexityEstimator}
    T = typeof(x)
    msg = "`complexity_normalized` not implemented for $C and input data of type $T."
    throw(ArgumentError(msg))
end
