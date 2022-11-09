# Entropies.jl Dev Docs

Good practices in developing a code base apply in every Pull Request. The [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop) is worth checking out for this.

## Adding a new `ProbabilitiesEstimator`

### Mandatory steps
1. Decide on the outcome space and how the estimator will map probabilities to outcomes.
2. Define your type and make it subtype `ProbabilitiesEstimator`.
3. Add a docstring to your type following the style of the docstrings of other estimators.
4. Implement dispatch for [`probabilities_and_outcomes`](@ref).
7. Implement dispatch for [`outcome_space`](@ref).
5.  Add your type to the list in the docstring of [`ProbabilitiyEstimator`](@ref).

### Optional steps
You may extend any of the following functions if there are potential performance benefits in doing so:

1. [`probabilities`](@ref). By default it calls `probabilities_and_outcomes` and returns the first value.
2. [`outcomes`](@ref). By default calls `probabilities_and_outcomes` and returns the second value.
3.  [`total_outcomes`](@ref). By default it returns the `length` of [`outcome_space`](@ref). This is the function that most typically has performance benefits if implemented explicitly, so most existing estimators extend it by default.