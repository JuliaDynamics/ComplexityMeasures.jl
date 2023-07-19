# New entropy: [information measure name here]

## Briefly describe the information measure type

What type of method is this? Is it a direct information measure (computes information measure from probabilities
that are estimated first), or an indirect estimator (computes information measure without estimating probabilities explicitly)?

## Checklist for development (optional)

Before finishing your PR, please have a look at the
project [devdocs](https://juliadynamics.github.io/ComplexityMeasures.jl/dev/devdocs/).

Ticking the boxes below will help us provide good feedback and speed up the review process.
Partial PRs are welcome too, and we're happy to help if you're stuck on something.

- [ ] The new information measure (estimator) subtypes `InformationMeasureDefinition` (`DiffInfoMeasureEst`).
- [ ] The new information measure has an informative docstring, which is referenced in
    `docs/src/information_measures.md`.
- [ ] Relevant sources are cited in the docstring.
- [ ] Dispatch for `information`, and `information_maximum` (if relevant/possible), is implemented.
- [ ] A runnable example is included in the `docs/src/example.md` file.
- [ ] Tests are implemented.

Before finalizing the PR, it is useful to check that

- [ ] The tests run successfully locally.
- [ ] The documentation build is successful locally, and looks good.
