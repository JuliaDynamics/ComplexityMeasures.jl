export KozachenkoLeonenko

"""
    KozachenkoLeonenko <: DifferentialInfoEstimator
    KozachenkoLeonenko(definition = Shannon(); w::Int = 0)

The `KozachenkoLeonenko` estimator [KozachenkoLeonenko1987](@cite)
computes the [`Shannon`](@ref) differential [`information`](@ref) of a multi-dimensional
[`StateSpaceSet`](@ref), with logarithms to the `base` specified in `definition`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `KozachenkoLeonenko` estimates
the [`Shannon`](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))]
```

using the nearest neighbor method from [KozachenkoLeonenko1987](@citet), as described in
[Charzyńska2015](@citet).

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

In contrast to [`Kraskov`](@ref), this estimator uses only the *closest* neighbor.

See also: [`information`](@ref), [`Kraskov`](@ref), [`DifferentialInfoEstimator`](@ref).
"""
struct KozachenkoLeonenko{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    w::Int
end

function KozachenkoLeonenko(definition = Shannon(); w = 0)
    return KozachenkoLeonenko(definition, w)
end

function information(est::KozachenkoLeonenko{<:Shannon}, x::AbstractStateSpaceSet{D}) where {D}
    (; w) = est

    N = length(x)
    ρs = maximum_neighbor_distances(x, w, 1)
    # The estimated entropy has "unit" [nats]
    h = 1/N * sum(log.(MathConstants.e, ρs .^ D)) +
        log(MathConstants.e, ball_volume(D)) +
        MathConstants.eulergamma +
        log(MathConstants.e, N - 1)
    return convert_logunit(h, ℯ, est.definition.base)
end
