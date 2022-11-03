## Contributor guidelines

- [ ] I verify that I have read the
    [devdocs](https://juliadynamics.github.io/Entropies.jl/dev/devdocs/).

## Description and references

Describe the method. How does it compute probabilities? Please cite appropriate scientific
literature. Outline any implementation details we should be aware of.

## Implementation

- [ ] The new estimator is a subtype of `ProbabilitiesEstimator`.
- [ ] Dispatch for `probabilities_and_outcomes` is implemented.
- [ ] Dispatch for `total_outcomes` is implemented (optional).

## Documentation

- [ ] The new estimator type has a docstring.
- [ ] The docstring explains how probabilities are computed.
- [ ] A level-two subheading, titled `MyNewEstimator` is included in
    `docs/src/probabilities.md`, and this section contains a `@docs` block
    referencing `MyNewEstimator`.
- [] A runnable example is included in the `docs/src/example.md` file.

## Testing

- [ ] Tests are implemented.
- [ ] Test cases with known input/output values which can be tested exactly are included.
    If not, why didn't you include such tests?

## Checklist before requesting a review

I confirmed that I have

- [ ] Self-reviewed my code.
- [ ] Locally ran the test suite with successful outcomes.
- [ ] Generated the documentation locally, and verified that it looks good.
- [ ] Cited all relevant method/code sources.
