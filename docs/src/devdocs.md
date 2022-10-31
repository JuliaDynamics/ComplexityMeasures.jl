# Entropies.jl Dev Docs

Good practices in developing a code base apply in every Pull Request. The [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop) is worth checking out for this.

## Adding a new `ProbabilitiesEstimator`
1. Define your type and make it subtype `ProbabilitiesEstimator`.
2. Add a docstring to your type following the style of the docstrings of other estimators.
3. Implement dispatch for [`probabilities`](@ref).
4. Implement dispatch for [`outcomes`](@ref).
5. Notice that [`probabilities_and_outcomes`](@ref) just calls the above two functions by default so it needs no implementation. Only in cases where there are performance gains, because some computations are shared between computing the probabilities and outcomes, you should implement the 2-in-1 function.
6. Implement [`total_outcomes`](@ref) if possible.
7. Implement [`missing_outcomes`](@ref) if possible.
8. Add your type to the list in the docstring of [`ProbabilitiyEstimator`](@ref).