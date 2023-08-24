using StateSpaceSets: AbstractStateSpaceSet, StateSpaceSet
using Neighborhood: KDTree, NeighborNumber, Theiler
using Neighborhood: bulksearch
using SpecialFunctions: digamma

export Goria

"""
    Goria <: DifferentialInfoEstimator
    Goria(measure = Shannon(); k = 1, w = 0)

The `Goria` estimator [Goria2005](@cite) computes the
[`Shannon`](@ref) differential
[`information`](@ref) of a multi-dimensional [`StateSpaceSet`](@ref),
with logarithms to the `base` specified in `definition`.

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Goria` estimates
the [Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

Specifically, let ``\\bf{n}_1, \\bf{n}_2, \\ldots, \\bf{n}_N`` be the distance of the
samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` to their
`k`-th nearest neighbors. Next, let the geometric mean of the distances be

```math
\\hat{\\rho}_k = \\left( \\prod_{i=1}^N \\right)^{\\dfrac{1}{N}}
```
[Goria2005](@citet)'s estimate of Shannon differential entropy is then

```math
\\hat{H} = m\\hat{\\rho}_k + \\log(N - 1) - \\psi(k) + \\log c_1(m),
```

where ``c_1(m) = \\dfrac{2\\pi^\\frac{m}{2}}{m \\Gamma(m/2)}`` and ``\\psi``
is the digamma function.
"""
struct Goria{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
end
function Goria(definition = Shannon(); k = 1, w = 0)
    return Goria(definition, k, w)
end

function information(est::Goria{<:Shannon}, x::AbstractStateSpaceSet{D}) where D
    (; k, w) = est
    N = length(x)

    tree = KDTree(x, Euclidean())
    ds = last.(bulksearch(tree, x, NeighborNumber(k), Theiler(w))[2])
    # The estimated entropy has "unit" [nats]
    h = D * log(prod(ds .^ (1 / N))) +
          log(N - 1) +
          log(c1(D)) -
          digamma(k)
    return convert_logunit(h, ℯ, est.definition.base)
end
c1(D::Int) = (2π^(D/2)) / (D* gamma(D/2))
