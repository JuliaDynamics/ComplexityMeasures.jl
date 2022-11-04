# New entropy: [entropy name here]

## Briefly describe the entropy type

What type of method is this? Is it a direct entropy (computes entropy from probabilities 
that are estimated first), or an indirect estimator (computes entropy without estimating probabilities explicitly)?

## Checklist for development (optional)

Before finishing your PR, please have a look at the
project[devdocs](https://juliadynamics.github.io/Entropies.jl/dev/devdocs/).

Ticking the boxes below will help us provide good feedback and speed up the review process.
Partial PRs are welcome too, and we're happy to help if you're stuck on something.

- [ ] The new entropy subtypes of `Entropy` or `IndirectEntropy`.
- [ ] The new entropy has an informative docstring, which is referenced in
    `docs/src/entropies.md`.
- [ ] Relevant sources are cited in the docstring.
- [ ] Dispatch for `entropy`, and `entropy_maximum` (if relevant/possible), is implemented.
- [ ] A runnable example is included in the `docs/src/example.md` file.
- [ ] Tests are implemented.

Before finalizing the PR, it is useful to check that

- [ ] The tests run successfully locally.
- [ ] The documentation build is successful locally, and looks good.
