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
function complexity end

"""
    complexity_normalized(c::ComplexityMeasure, x) → m ∈ [a, b]

Estimate the [`complexity`](@ref) measure `c` for [input data](@ref input_data) `x`,
normalized to the interval `[a, b]`, where `[a, b]` depends on `c`.
"""
function complexity_normalized(c::C, args...; kwargs...) where {C <: ComplexityMeasure}
    throw(ArgumentError("complexity_normalized not implemented for $C."))
end

include("complexity_measures/complexity_measures.jl")
