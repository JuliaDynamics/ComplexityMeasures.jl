# ComplexityMeasures.jl Dev Docs

Good practices in developing a code base apply in every Pull Request. The [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop) is worth checking out for this.

## Adding a new `ProbabilitiesEstimator`

### Mandatory steps

1. Decide on the outcome space and how the estimator will map probabilities to outcomes.
2. Define your type and make it subtype [`ProbabilitiesEstimator`](@ref).
3. Add a docstring to your type following the style of the docstrings of other estimators.
4. If suitable, the estimator may be able to operate based on [`Encoding`](@ref)s. If so, it is preferred to implement an `Encoding` subtype and extend the methods [`encode`](@ref) and [`decode`](@ref). This will allow your probabilities estimator to be used with a larger span of entropy and complexity methods without additional effort. Have a look at the file defining [`SymbolicPermutation`](@ref) for an idea of how this works.
5. Implement dispatch for [`probabilities_and_outcomes`](@ref) and your probabilities estimator type.
6. Implement dispatch for [`outcome_space`](@ref) and your probabilities estimator type.
7. Add your probabilities estimator type to the table list in the documentation page of probabilities. If you made an encoding, also add it to corresponding table in the encodings section.

### Optional steps

You may extend any of the following functions if there are potential performance benefits in doing so:

1. [`probabilities`](@ref). By default it calls `probabilities_and_outcomes` and returns the first value.
2. [`outcomes`](@ref). By default calls `probabilities_and_outcomes` and returns the second value.
3. [`total_outcomes`](@ref). By default it returns the `length` of [`outcome_space`](@ref). This is the function that most typically has performance benefits if implemented explicitly, so most existing estimators extend it by default.

### Tests

You also need to add tests for **all** functions that you **explicitly** extended.
Non-extended functions do not need to be tested.

## Adding a new `DifferentialEntropyEstimator`

### Mandatory steps

1. Define your type and make it subtype [`DifferentialEntropyEstimator`](@ref).
2. Add a docstring to your type following the style of the docstrings of other estimators.
    This docstring should contain the formula(s)/integral(s) which it estimates, and a
    reference to relevant [`EntropyDefinition`](@ref)(s).
3. Implement dispatch for [`entropy`](@ref) with the relevant [`EntropyDefinition`](@ref).
    If your estimator works for multiple entropies, implement one method for
    [`entropy`](@ref) for each of them.

### Tests

You need to add tests verifying that your estimator actually convergences, within
some reasonable tolerance (that you define), to the true entropy of data from
some known distribution. Have a look in the tests for existing estimators for
inspiration (you can just copy-paste one of the existing tests, or make them
more elaborate if you want to).

## Adding a new `EntropyDefinition`

### Mandatory steps

1. Define your entropy definition type and make it subtype [`EntropyDefinition`](@ref).
2. Implement dispatch for [`entropy`](@ref)`(def::YourType, p::Probabilities)`
3. Add a docstring to your type following the style of the docstrings of other entropy
    definitions, and should include the mathematical definition of the entropy.
4. Add your entropy definition type to the list of definitions in the
    `docs/src/entropies.md` documentation page.
5. Add a reference to your entropy definition in the docstring for
    [`EntropyDefinition`](@ref).

### Optional steps

1. If the maximum value of your entropy type is analytically computable for a probability
    distribution with a known number of elements, implementing dispatch for
    [`entropy_maximum`](@ref) automatically enables [`entropy_normalized`](@ref) for your
    type.

### Tests

You also need to add tests for **all** functions that you **explicitly** extended.
Non-extended functions do not need to be tested.
