export InformationFluctuation

"""
    InformationFluctuation <: InformationMeasure
    InformationFluctuation(; definition = Shannon())

The information fluctuation quantifies the standard deviation of the information content of the states 
``\\omega_i`` around some summary statistic ([`InformationMeasure`](@ref)) of a PMF. Specifically, given some 
outcome space ``\\Omega`` with outcomes ``\\omega_i \\in \\Omega`` 
and a probability mass function ``p(\\Omega) = \\{ p(\\omega_i) \\}_{i=1}^N``, it is defined as

```math
\\sigma_I_Q(p) := \\sqrt{\\sum_{i=1}^N p_i(I_Q(p_i) - F_Q)^2}
```

where ``I_Q(p_i)`` is the [`information_content`](@ref) of the i-th outcome with respect to the information 
measure ``F_Q`` (controlled by `definition`).

## Compatible with

- [`Shannon`](@ref)
- [`Tsallis`](@ref)
- [`Curado`](@ref)
- [`StretchedExponential`](@ref)
- [`ShannonExtropy`](@ref)

If `definition` is the [`Shannon`](@ref) entropy, then we recover the 
[Shannon-type "information fluctuation complexity"](https://en.wikipedia.org/wiki/Information_fluctuation_complexity) 
from [Bates1993](@cite). 

## Properties 

Then the information fluctuation is zero for PMFs with only a single non-zero element, or 
for the uniform distribution.

## Examples

```julia
using ComplexityMeasures
using Random; rng = Xoshiro(55543)

# Information fluctuation for a time series encoded by ordinal patterns
x = rand(rng, 10000)
def = Tsallis(q = 2) # information measure definition
pest = RelativeAmount() # probabilities estimator
o = OrdinalPatterns(m = 3) # outcome space / discretization method
information(InformationFluctuation(definition = def), pest, o, x)
```

!!! note "Potential for new research" 
    As far as we know, using other information measures besides Shannon entropy for the 
    fluctuation complexity hasn't been explored in the literature yet. Our implementation, however, allows for it.
    We're currently writing a paper outlining the generalizations to other measures. For now, we verify 
    correctness of the measure through numerical examples in our test-suite.
"""
struct InformationFluctuation{M <: InformationMeasure, I <: Integer} <: InformationMeasure
    definition::M
    base::I

    function InformationFluctuation(; definition::D = Shannon(base = 2), base::I = 2) where {D, I}
        if D isa InformationFluctuation
            throw(ArgumentError("Cannot use `InformationFluctuation` as the summary statistic for `InformationFluctuation`. Please select some other information measures, like `Shannon`."))
        end
        return new{D, I}(definition, base)
    end
end

# Fluctuation complexity is zero when p_i = 1/N or when p = (1, 0, 0, ...).
function information(e::InformationFluctuation, probs::Probabilities)
    def = e.definition
    non0_probs = Iterators.filter(!iszero, vec(probs))
    h = information(def, probs)
    return sqrt(sum(pᵢ * (self_information(def, pᵢ, length(probs)) - h)^2 for pᵢ in non0_probs))
end

function information_normalized(e::InformationFluctuation, probs::Probabilities)
    def = e.definition
    non0_probs = Iterators.filter(!iszero, vec(probs))
    h = information(def, probs)
    info_fluct = sqrt(sum(pᵢ * (self_information(def, pᵢ, length(probs)) - h)^2 for pᵢ in non0_probs))
    return info_fluct / h
end

# The maximum is not generally known.