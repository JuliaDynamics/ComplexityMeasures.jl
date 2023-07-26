export MillerMadow

"""
    MillerMadow <: DiscreteInfoEstimator
    MillerMadow(measure::Shannon = Shannon())

The `MillerMadow` estimator computes the [`Shannon`](@ref) discrete entropy according
to Miller (1955)[^Miller1955].

# Description

The Miller-Madow estimator of Shannon entropy is given by

```math
H_S^{MM} = H_S^{plugin} + \\dfrac{m - 1}{2N},
```

where `H_{plugin}` is the Shannon entropy estimated using the [`PlugIn`](@ref) estimator,
`m` is the number of bins with nonzero probability (as defined in Paninski,
2003[^Paninski2003]), and `N` is the number of observations.

[^Miller1955]:
    Miller, G. (1955). Note on the bias of information estimates. Information theory in
    psychology: Problems and methods.

[^Paninski2003]:
    Paninski, L. (2003). Estimation of entropy and mutual information. Neural computation,
    15(6), 1191-1253.
"""
struct MillerMadow{I <: InformationMeasure} <: DiscreteInfoEstimator{I}
    measure::I
end
MillerMadow() = MillerMadow(Shannon())

function information(hest::MillerMadow{<:Shannon}, pest::ProbabilitiesEstimator, x)
    probs = allprobabilities(pest, x)
    # Estimate of the number of bins with nonzero pN-probability; here estimated as
    # in Paninski (2003)
    m̂ = count(probs .> 0.0)

    h_naive = information(PlugIn(hest.measure), pest, x)
    return h_naive + (m̂ - 1)/(2 * length(x))
end
