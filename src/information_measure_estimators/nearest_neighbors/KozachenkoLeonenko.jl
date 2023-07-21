export KozachenkoLeonenko

"""
    KozachenkoLeonenko <: DifferentialInfoEstimator
    KozachenkoLeonenko(measure = Shannon(); w::Int = 0, base = 2)

The `KozachenkoLeonenko` estimator computes the [`Shannon`](@ref) differential
[`information`](@ref) of a multi-dimensional [`StateSpaceSet`](@ref) in the given `base`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `KozachenkoLeonenko` estimates
the [Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))]
```

using the nearest neighbor method from Kozachenko &
Leonenko (1987)[^KozachenkoLeonenko1987], as described in Charzyńska and
Gambin[^Charzyńska2016].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

In contrast to [`Kraskov`](@ref), this estimator uses only the *closest* neighbor.


See also: [`information`](@ref), [`Kraskov`](@ref), [`DifferentialInfoEstimator`](@ref).

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy
    estimator with applications in systems biology. InformationMeasure, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of
    the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
struct KozachenkoLeonenko{I <: InformationMeasure, B} <: NNDifferentialInfoEstimator{I}
    measure::I
    w::Int
    base::B
end

function KozachenkoLeonenko(measure = Shannon(); w = 0, base = 2)
    return KozachenkoLeonenko(measure, w, base)
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
    return convert_logunit(h, ℯ, est.base)
end
