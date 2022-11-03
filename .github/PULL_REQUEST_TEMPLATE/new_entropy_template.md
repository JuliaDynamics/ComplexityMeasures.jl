## Contributor guidelines

- [ ] I verify that I have read the
    [devdocs](https://juliadynamics.github.io/Entropies.jl/dev/devdocs/).

## Description and references

What is this new entropy called? Please cite appropriate scientific
literature. Outline any implementation details we should be aware of.

## What type of entropy (estimator) is this?

- [ ] Computes entropy directly from probabilities that are explicitly estimated.
- [ ] Computes entropy indirectly, in way that doesn't explicitly estimate probabilities.

## Implementation

- [ ] The new entropy is a subtype of `Entropy` or `IndirectEntropy`.
- [ ] Dispatch for `entropy` is implemented.
- [ ] Dispatch for `entropy_maximum` is implemented. If not, why not?

## Documentation

- [ ] The new entropy type has a docstring.
- [ ] The docstring shows, using LaTeX notation, what formula the entropy computes.
- [ ] A level-two subheading, titled `MyNewEntropy` is included in
    `docs/src/entropies.md`, and this section contains a `@docs` block
    referencing `MyNewEntropy`.
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
