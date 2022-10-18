export ComplexityMeasure
export complexity
export complexity_normalized

"""
    ComplexityMeasure

Abstract type for (entropy-like) complexity measures.
"""
abstract type ComplexityMeasure end

"""
    complexity(c::ComplexityMeasure, x)

Estimate the complexity measure `c` for input data `x`, where `c` can be any of the
following measures:

- [`ReverseDispersion`](@ref).

"""
function complexity end

"""
    complexity_normalized(c::ComplexityMeasure, x) → m ∈ [a, b]

Estimate the normalized complexity measure `c` for input data `x`, where `c` can
can be any of the following measures:

- [`ReverseDispersion`](@ref).

The potential range `[a, b]` of the output value depends on `c`. See the documentation
strings for the individual measures to get the normalized ranges.
"""
function complexity_normalized end

include("complexity_measures/complexity_measures.jl")
