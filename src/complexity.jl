export ComplexityMeasure
export complexity
export complexity_normalized

"""
    ComplexityMeasure

Supertype for (entropy-like) complexity measures.
"""
abstract type ComplexityMeasure end

"""
    complexity(c::ComplexityMeasure, x)

Estimate the complexity measure `c` for [input data](@ref input_data) `x`, where `c` can
be any of the following measures:

- [`ReverseDispersion`](@ref).
"""
function complexity(c::C, x) where C <: ComplexityMeasure
    T = typeof(x)
    msg = "`complexity` not implemented for $C and input data of type $T."
    throw(ArgumentError(msg))
end

"""
    complexity_normalized(c::ComplexityMeasure, x) → m ∈ [a, b]

Estimate the [`complexity`](@ref) measure `c` for [input data](@ref input_data) `x`,
normalized to the interval `[a, b]`, where `[a, b]` depends on `c`.
"""
function complexity_normalized(c::C, x) where {C <: ComplexityMeasure}
    T = typeof(x)
    msg = "`complexity_normalized` not implemented for $C and input data of type $T."
    throw(ArgumentError(msg))
end

include("complexity_measures/complexity_measures.jl")
