
# New probabilities estimator: [estimator name here]

What type of estimator is this describe what type of estimator this is. How does it compute
probabilities?

## Checklist for development (optional)

Before finishing your PR, please have a look at the
project [devdocs](https://juliadynamics.github.io/ComplexityMeasures.jl/dev/devdocs/).

Ticking the boxes below will help us provide good feedback and speed up the review process.
Partial PRs are welcome too, and we're happy to help if you're stuck on something.

- [ ] The new estimator subtypes `ProbabilitiesEstimator`
- [ ] The new estimator has an informative docstring, which is referenced in
    `docs/src/probabilities.md`.
- [ ] Relevant sources are cited in the docstring.
- [ ] Dispatch for `probabilities_and_outcomes`, and `total_outcomes`
    (if relevant/possible), is implemented.
- [ ] A runnable example is included in the `docs/src/example.md` file.
- [ ] Tests are implemented.

Before finalizing the PR, it is useful to check that

- [ ] The tests run successfully locally.
- [ ] The documentation build is successful locally, and looks good.
