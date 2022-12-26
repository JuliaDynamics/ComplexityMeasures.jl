# CHANGELOG

Changelog is kept with respect to version 0.11 of Entropies.jl.

## 2.0

The API for Entropies.jl has been completely overhauled. Major changes are:

- Common generic interfaces `entropy`, `entropy_normalized` and `maximum` (maximum entropy) that dispatches on different types of entropies (e.g `Renyi()` `Shannon()`, `Tsallis()`).
- Convenience functions for common entropies, such as permutation entropy and dispersion entropy.
- No more deprecation warnings for using the old keyword `α` for Renyi entropy.
- The `base` of the entropy is now a field of the `EntropyDefinition` type, not the estimator.
    You'll now have to do `entropy(Shannon(; base = 2), est, x)`.
- An entirely new section of entropy-like complexity measures, such as the reverse dispersion entropy.
- Many new estimators, such as `SpatialPermutation` and `PowerSpectrum`.
- Check the online documentation for a comprehensive overview of the changes.

### Bug fixes

- The `KozachenkoLeonenko` estimator now correctly fixes its neighbor search to the
    *closest* neighbor only, and its constructor does no longer accept `k` as an input. It also uses correct scaling factor and adapts to dimension.
- Using a logarithm `base` different from `MathConstants.e` now yields correct results
    for `Kraskov` and `KozachenkoLeonenko`.

## main
* New probability estimator `SpatialSymbolicPermutation` suitable for computing spatial permutation entropies
* Introduce Tsallis entropy.

## 1.2
* Added dispersion entropy.

## 1.1
* Introduce convenience function `permentropy`.
* Several type instabilities fixed.

## 1.0
No actual changes, just first major version release.

## 0.12
* Nearest neighbor searches now use Neighborhood.jl and the Theiler window properly.

## 0.11.1
* `probabilities(data, n::Int)` now uses a rectangular binning of `n` bins for each dimension. Before, while not documented as possible in the public API, using integer `n` would take it as the bin size.
