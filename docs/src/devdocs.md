# Entropies.jl Dev Docs

Good practices in developing a code base apply in every Pull Request. The [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop) is worth checking out for this.

## Adding a new `ProbabilitiesEstimator`
1. Define your type and make it subtype `ProbabilitiesEstimator`.
2. Add a docstring to your type following the style of the docstrings of other estimators.
3. Implement dispatch for [`probabilities_and_outcomes`](@ref).
4. Notice that [`probabilities`](@ref) by default calls `probabilities_and_outcomes` and returns the first value, so it needs no implementation. Only in cases where there are performance gains, because e.g. additional costly operations are necessary to compute the outcomes, should a dedicated version for `probabilities` be implemented. This is e.g. what we do for `ValueHistogram`.
5. Similarly, [`outcomes`](@ref) by default calls `probabilities_and_outcomes` and returns the second value. In exceptionally rare cases of needing to squeeze all possible performance out of calculating outcomes, the `outcomes` function may be explicitly extended.
6. Implement [`total_outcomes`](@ref) if possible. This also gives [`missing_outcomes`](@ref) for free. If the estimator can provide `total_outcomes` without knowledge of input data, the implement `total_outcomes(est)` directly instead of `total_outcomes(x, est)`.
7. Add your type to the list in the docstring of [`ProbabilitiyEstimator`](@ref).