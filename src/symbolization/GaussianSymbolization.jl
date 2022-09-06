using Statistics, QuadGK

export GaussianSymbolization

"""
    GaussianSymbolization(; n_categories::Int = 5)

A symbolization scheme where the elements of `x`, which is the time series
``x = \\{x_j, j = 1, 2, \\ldots, N \\}``, is to a new time series
``y = \\{y_j, j = 1, 2, \\ldots, N \\}`` where ``y_j \\in (0, 1)`` and

```math
y_j = \\dfrac{1}{\\sigma\\sqrt{2\\pi}} \\int_{x = -\\infty}^{x_j} \\exp{\\dfrac{-{(x-\\mu)}^2}{2{\\sigma}^2}} dx,
```

where ``\\mu`` and ``\\sigma`` are the mean and standard deviations of `x`.

# Usage

    symbolize(x::AbstractVector, s::GaussianSymbolization)

Map the elements of `x` to a symbol time series according to the Gaussian symbolization
scheme `s`.

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = [0.1, 0.4, 0.7, -2.1, 8.0, 0.9, -5.2];

julia> Entropies.symbolize(x, GaussianSymbolization(5))
7-element Vector{Int64}:
 3
 3
 3
 2
 5
 3
 1
```

See also: [`symbolize`](@ref).
"""
Base.@kwdef struct GaussianSymbolization{I <: Integer}
    n_categories::I
end

g(xᵢ, μ, σ) = exp((-(xᵢ - μ)^2)/(2σ^2))

"""
    map_to_category(yⱼ, n_categories)

Map the value `yⱼ ∈ (0, 1)` to an integer `[1, 2, …, n_categories]`.
"""
function map_to_category(yⱼ, n_categories)
    zⱼ = ceil(Int, yⱼ * (n_categories - 1) + 1/2)
    return zⱼ
end

function symbolize(x::AbstractVector, s::GaussianSymbolization)
    σ = Statistics.std(x)
    μ = Statistics.mean(x)

    # We only need the value of the integral (not the error), so
    # index first element returned from quadgk
    k = 1/(σ*sqrt(2π))
    yⱼs = [k * first(quadgk(x -> g(x, μ, σ), -Inf, xᵢ)) for xᵢ in x]

    return map_to_category.(yⱼs, s.n_categories)
end
