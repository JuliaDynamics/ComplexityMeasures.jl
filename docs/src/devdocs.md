# Entropies.jl Dev Docs

## Adding a new `ProbabilitiesEstimator`
1. Define your type and make it subtype `ProbabilitiesEstimator`.
2. Add a docstring to your type following the style of the docstrings of other estimators.
3. Implement dispatch for [`probabilities_and_outcomes`](@ref). By default, `probabilities, outcomes` call the above function and provide the necessary output. Notice that `probabilities` should only implemented if there are performance benefits to do so, i.e., calculating the `outcomes` explicitly requires additional computational cost.
4. Add your type to the list in the docstring of [`ProbabilitiyEstimator`](@ref).