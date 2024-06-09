export FluctuationComplexity

"""
    FluctuationComplexity <: InformationMeasure
    FluctuationComplexity(; definition = Shannon())

The "fluctuation complexity" quantifies the standard deviation of the information content of the states 
``\\omega_i`` around some summary statistic ([`InformationMeasure`](@ref)) of a PMF. Specifically, given some 
outcome space ``\\Omega`` with outcomes ``\\omega_i \\in \\Omega`` 
and a probability mass function ``p(\\Omega) = \\{ p(\\omega_i) \\}_{i=1}^N``, it is defined as

```math
\\sigma_I_Q(p) := \\sqrt{\\sum_{i=1}^N p_i(I_Q(p_i) - H_*)^2}
```

where ``I_Q(p_i)`` is the [`self_information`](@ref) of the i-th outcome with respect to the information 
measure of type ``Q`` (controlled by `definition`).

## Compatible with

- [`Shannon`](@ref)
- [`Tsallis`](@ref)
- [`Curado`](@ref)
- [`ShannonExtropy`](@ref)

## Properties 

If `definition` is the [`Shannon`](@ref) entropy, then we recover the 
[Shannon-type information fluctuation complexity](https://en.wikipedia.org/wiki/Information_fluctuation_complexity) 
from [Bates1993](@cite). Then the fluctuation complexity is zero for PMFs with only a single non-zero element, or 
for the uniform distribution.

If `definition` is not Shannon entropy, then the properties of the measure varies, and does not necessarily share the 
properties [Bates1993](@cite). 

!!! note "Potential for new research" 
    As far as we know, using other information measures besides Shannon entropy for the 
    fluctuation complexity hasn't been explored in the literature yet. Our implementation, however, allows for it.
    We're currently writing a paper outlining the generalizations to other measures. For now, we verify 
    correctness of the measure through numerical examples in out test-suite.
"""
struct FluctuationComplexity{M <: InformationMeasure, I <: Integer} <: InformationMeasure
    definition::M
    base::I

    function FluctuationComplexity(; definition::D = Shannon(base = 2), base::I = 2) where {D, I}
        if D isa FluctuationComplexity
            throw(ArgumentError("Cannot use `FluctuationComplexity` as the summary statistic for `FluctuationComplexity`. Please select some other information measures, like `Shannon`."))
        end
        return new{D, I}(definition, base)
    end
end

# Fluctuation complexity is zero when p_i = 1/N or when p = (1, 0, 0, ...).
function information(e::FluctuationComplexity, probs::Probabilities)
    def = e.definition
    h = information(def, probs)
    non0_probs = Iterators.filter(!iszero, vec(probs))
    logf = log_with_base(e.base)
    return sqrt(sum(pᵢ * (-logf(pᵢ) - h) ^ 2 for pᵢ in non0_probs))
end

# The maximum is not generally known.