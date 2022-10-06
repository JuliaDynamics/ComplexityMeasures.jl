"""
    SigmoidSymbolization(; c::Int = 3)

A symbolization scheme where the elements of `x` are symbolized into `c` distinct integer
categories using the normal cumulative distribution function (NCDF).

## Algorithm

Assume we have a univariate time series ``X = \\{x_i\\}_{i=1}^N``. `SigmoidSymbolization`
first maps each ``x_i`` to a new real number ``y_i \\in [0, 1]`` by using the normal
cumulative distribution function (CDF), ``x_i \\to y_i : y_i = \\dfrac{1}{ \\sigma
    \\sqrt{2 \\pi}} \\int_{-\\infty}^{x_i} e^{(-(x_i - \\mu)^2)/(2 \\sigma^2)} dx``,
where ``\\mu`` and ``\\sigma`` are the empirical mean and standard deviation of ``X``.

Next, each ``y_i`` is linearly mapped to an integer
``z_i \\in [1, 2, \\ldots, c]`` using the map
``y_i \\to z_i : z_i = R(y_i(c-1) + 0.5)``, where ``R`` indicates rounding up to the
nearest integer. This procedure subdivides the interval ``[0, 1]`` into ``c``
different subintervals that form a covering of ``[0, 1]``, and assigns each ``y_i`` to one
of these subintervals. The original time series ``X`` is thus transformed to a symbol time
series ``S = \\{ s_i \\}_{i=1}^N``, where ``s_i \\in [1, 2, \\ldots, c]``.

# Usage

    symbolize(x::AbstractVector, s::SigmoidSymbolization)

Map the elements of `x` to a symbol time series according to the Sigmoid symbolization
scheme `s`.

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = [0.1, 0.4, 0.7, -2.1, 8.0, 0.9, -5.2];

julia> Entropies.symbolize(x, SigmoidSymbolization(c = 5))
7-element Vector{Int64}:
 3
 3
 3
 2
 5
 3
 1
```
