export ComplexityEstimator
export complexity
export complexity_normalized

"""
    ComplexityEstimator

Supertype for (entropy-like) complexity measures.
"""
abstract type ComplexityEstimator end

"""
    complexity(c::ComplexityEstimator, x)

Estimate the complexity measure `c` for [input data](@ref input_data) `x`, where `c` can
be any of the following measures:

- [`ReverseDispersion`](@ref).
- [`ApproximateEntropy`](@ref).
- [`SampleEntropy`](@ref).
- [`MissingDispersionPatterns`](@ref).
"""
function complexity(c::C, x) where C <: ComplexityEstimator
    T = typeof(x)
    msg = "`complexity` not implemented for $C and input data of type $T."
    throw(ArgumentError(msg))
end

"""
    complexity_normalized(c::ComplexityEstimator, x) → m ∈ [a, b]

The same as [`complexity`](@ref), but the result is normalized to the interval `[a, b]`,
where `[a, b]` depends on `c`.
"""
function complexity_normalized(c::C, x) where {C <: ComplexityEstimator}
    T = typeof(x)
    msg = "`complexity_normalized` not implemented for $C and input data of type $T."
    throw(ArgumentError(msg))
end
