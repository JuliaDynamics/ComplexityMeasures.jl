# [Convenience functions](@ref convenience)

We provide a few convenience functions for widely used names for entropy or "entropy-like" quantities. Other arbitrary specialized convenience functions can easily be defined in a couple lines of code.

We emphasize that these functions really aren't anything more than
2-lines-of-code wrappers that call [`information`](@ref) with the appropriate
[`OutcomeSpace`](@ref) and [`InformationMeasure`](@ref).

```@docs
entropy_permutation
entropy_wavelet
entropy_dispersion
entropy_approx
entropy_sample
```
