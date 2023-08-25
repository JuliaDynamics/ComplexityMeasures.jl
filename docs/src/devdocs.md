# ComplexityMeasures.jl Dev Docs

Good practices in developing a code base apply in every Pull Request. The [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop) is worth checking out for this.

All PRs contributing new functionality must be well tested and well documented. You only need to add tests for methods that you **explicitly** extended.

## Adding a new `OutcomeSpace`

### Mandatory steps

1. Decide on the outcome space and how the estimator will map probabilities to outcomes.
2. Define your type and make it subtype [`OutcomeSpace`](@ref).
3. Add a docstring to your type following the style of the docstrings of other estimators.
4. If suitable, the estimator may be able to operate based on [`Encoding`](@ref)s. If so,
    it is preferred to implement an `Encoding` subtype and extend the methods
    [`encode`](@ref) and [`decode`](@ref). This will allow your outcome space to be used
    with a larger span of entropy and complexity methods without additional effort.
    Have a look at the file defining [`OrdinalPatterns`](@ref) for an idea of how this
    works.

If your new outcome space is counting-based, then

5. Implement dispatch for [`counts`](@ref) for your [`OutcomeSpace`](@ref) type.
6. Implement dispatch for [`counts_and_outcomes`](@ref) for your [`OutcomeSpace`](@ref) type.

If your new outcome space is not counting-based, then

7. Implement dispatch for [`probabilities`](@ref) for your [`OutcomeSpace`](@ref) type.
8. Implement dispatch for [`probabilities_and_outcomes`](@ref) for your
    [`OutcomeSpace`](@ref) type.

Finally,

9. Implement dispatch for [`outcome_space`](@ref) and your [`OutcomeSpace`](@ref) type.
    The return value of `outcome_space` must be sorted (as in the default behavior of
    `sort`, in ascending order).
10. Add your outcome space type to the table list in the documentation page of outcome
    space. If you made an encoding, also add it to corresponding table in the encodings
    section.

### Optional steps

1. [`outcomes`](@ref). By default calls [`probabilities_and_outcomes`](@ref) and returns
    the second value.
2. [`total_outcomes`](@ref). By default it returns the `length` of [`outcome_space`](@ref).
    This is the function that most typically has performance benefits if implemented
    explicitly, so most existing estimators extend it by default.

## Adding a new [`ProbabilitiesEstimator`](@ref)

### Mandatory steps 

1. Define your type and make it subtype [`ProbabilitiesEstimator`](@ref).
2. Add a docstring to your type following the style of the docstrings of other
    [`ProbabilitiesEstimator`](@ref)s.
3. Implement dispatch for [`probabilities_and_outcomes`](@ref) for your
    [`ProbabilitiesEstimator`](@ref) type.
4. Implement dispatch for [`allprobabilities_and_outcomes`](@ref) for your
    [`ProbabilitiesEstimator`](@ref) type.
5. Add your new [`ProbabilitiesEstimator`](@ref) type to the list of probabilities
    estimators in the probabilities estimators documentation section.

### Optional steps

1. Implement dispatch for [`probabilities`](@ref) for your [`ProbabilitiesEstimator`](@ref)
    if doing so leads to more efficient code.
2. Implement dispatch for [`allprobabilities`](@ref) for your [`ProbabilitiesEstimator`](@ref)
    if doing so leads to more efficient code.

## Adding a new `InformationMeasureEstimator`

The type implementation should follow the declared API of [`InformationMeasureEstimator`](@ref). If the type is a discrete measure, then extend `information(e::YourType, p::Probabilities)`. If it is a differential measure, then extend `information(e::YourType, x::InputData)`.

```@docs
InformationMeasureEstimator
```

## Adding a new `InformationMeasure`

This amounts to adding a new definition of an information measure, not an estimator. It de-facto means adding a method for the discrete Plug-In estimator.

### Mandatory steps

1. Define your information measure definition type and make it subtype [`InformationMeasure`](@ref).
2. Implement dispatch for [`information`](@ref)`(def::YourType, p::Probabilities)`. This is the Plug-In estimator for the discrete measure.
3. Add a docstring to your type following the style of the docstrings of other information
    measure definitions, and should include the mathematical definition of the measure.
4. Add your information measure definition type to the list of definitions in the
    `docs/src/information_measures.md` documentation page.
5. Add a reference to your information measure definition in the docstring for
    [`InformationMeasure`](@ref).

### Optional steps

1. If the maximum value of your information measure type is analytically computable for a
    probability distribution with a known number of elements, implementing dispatch for
    [`information_maximum`](@ref) automatically enables [`information_normalized`](@ref)
    for your type.
