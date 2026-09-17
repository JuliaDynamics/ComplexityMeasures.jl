# Remove these when PR is done
using ComplexityMeasures
import ComplexityMeasures: CountBasedOutcomeSpace, outcome_space, total_outcomes, counts_and_outcomes, codify, encode

"""
    SequentialSlopes <: CountBasedOutcomeSpace
    SequentialSlopes(m::Int, thresholds::AbstractVector)
    SequentialSlopes(m::Int; γ = 1, γ_u = γ, γ_d = -γ_u, δ = 0.001)

An outcome space based on the slopes joining consecutive values of a
timeseries, used to compute the slope entropy of [CuestaFrau2019](@cite)
and its asymmetric-threshold generalization of [Kouka2024](@cite).
The outcome space generalises to an arbitrary number of positive and negative slopes.

`SlopesOfDifferences` operates on scalar (`Real`-valued), univariate
timeseries with at least `m + 1` points.

## Description

For a univariate timeseries `x`, the differences ``d_i = x_{i+1} - x_i``
are symbolized according to a sorted vector of `thresholds`
``t_1 < t_2 < \\ldots < t_k``, which partition the real line into
``k+1`` left-closed-right-open bins ``[t_i, t_{i+1})`` (following the same
binning convention as e.g. [`ValueBinning`](@ref)). Each bin is mapped to
an integer symbol so that the bin containing (approximately) zero-valued
differences is mapped to `0`.

With the keyword constructor, `thresholds = [γ_d, -δ, δ, γ_u]`, so that,
using the default symmetric case `γ_u = -γ_d = γ`, differences are
symbolized into the classic 5-letter alphabet ``\\{-2,-1,0,1,2\\}`` of
[CuestaFrau2019](@cite):

```math
s_i = \\begin{cases}
 2  & \\text{if } d_i \\geq \\gamma_u \\\\
 1  & \\text{if } \\delta \\leq d_i < \\gamma_u \\\\
 0  & \\text{if } -\\delta \\leq d_i < \\delta \\\\
-1  & \\text{if } \\gamma_d \\leq d_i < -\\delta \\\\
-2  & \\text{if } d_i < \\gamma_d
\\end{cases}
```

Here `δ` is the small threshold separating "flat" from "moderate" slopes,
and `γ` is the large threshold separating "moderate" from "steep" slopes.
Passing `γ_u` and `γ_d` independently (instead of via the symmetric `γ`)
reproduces the **asymmetric** thresholding scheme of [Kouka2024](@cite),
in which positive and negative gradients are classified using different
steep thresholds.

The general two-argument constructor `SlopesOfDifferences(m, thresholds)`
allows an arbitrary sorted vector of thresholds, generalizing beyond
the 5-symbol alphabet (e.g. to add extra interval parameters).

## Outcome space

`SlopesOfDifferences` is a [`CountBasedOutcomeSpace`](@ref). The outcome
space `Ω` consists of the length-`m` words formed by sliding a window of
length `m` (step `1`) over the symbolized difference sequence, i.e. the
same embedding scheme used for e.g. [`OrdinalPatterns`](@ref). Each
outcome is thus an `NTuple{m, Int} and `total_outcomes` is `(length(thresholds) + 1)^m`
(`5^m` in the default, 5-symbol case).
"""
struct SequentialSlopes{T<:Real} <: CountBasedOutcomeSpace
    m::Int
    # thresholds contains all the `γ, δ` difference thresholds,
    # while allowing generalizability to as many thresholds as we want.
    thresholds::Vector{T}
    zero_integer::Int # what to subtract to make symbols symmetric around 0
end

function SequentialSlopes(m::Int, thresholds::AbstractVector)
    issorted(thresholds) || throw(ArgumentError("thresholds must be sorted."))
    length(unique(thresholds)) ≠ length(thresholds) && throw(ArgumentError("thresholds must be unique."))
    return SequentialSlopes(m, thresholds, length(thresholds)÷2)
end

function SequentialSlopes(m::Int; γ = 1.0, γ_u = γ, γ_d = -γ_u, δ = 0.001)
    return SequentialSlopes(m, [γ_d, -δ, δ, γ_u])
end

function outcome_space(o::SequentialSlopes)
    alphabet = (1:(length(o.thresholds)+1)) .- (o.zero_integer + 1)
    Ω = vec(collect(Iterators.product(ntuple(_ -> alphabet, o.m)...)))
    sort!(Ω) # TODO: Is this `sort!` needed?
    return Ω
end

# Performance extension
total_outcomes(o::SequentialSlopes) = (length(o.thresholds) + 1)^o.m

function slope_symbol(o::SequentialSlopes, d::Real)
    # index is defined so its minimum value is 0
    # and maximum value is possible slope classes minus 1
    index = searchsortedlast(o.thresholds, d)
    # which is then transformed to the convention of the paper
    return index - o.zero_integer
end

function codify(o::SequentialSlopes, x::AbstractVector{<:Real})
    # d = diff(x)
    # symbols = encode.(Ref(o.encoding), d)
    indices = 1:length(x)-1
    symbols = map(i -> slope_symbol(o, x[i+1] - x[i]), indices)
    return symbols
end

function counts_and_outcomes(o::SequentialSlopes, x::AbstractVector{<:Real})
    symbols = codify(o, x)
    words = ComplexityMeasures.embed(symbols, o.m, 1)
    cts = ComplexityMeasures.fasthist!(words) # this sorts the words
    outs = unique!(words) # therefore, outcomes are the sorted patterns.
    c = Counts(cts, (outs,))
    return c, outcomes(c)
end
