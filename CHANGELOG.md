# CHANGELOG

Changelog is kept with respect to version 0.11 of Entropies.jl. From version v2.0 onwards, this package has been renamed to ComplexityMeasures.jl.

## 3.7

- Updated to StateSpaceSets.jl v2.0
- Fixed a bug in `codify` with `StateSpaceSet`. Now it does exactly as described in the docstring.

## 3.6

- New information measure: `FluctuationComplexity`.

## 3.5

- New multiscale API.
- New spatial outcome space: `SpatialBubbleSortSwaps`.
- A script in the documentation now calculates explicitly the total possible complexity measures one can estimate with ComplexityMeasures.jl. For version 3.5 this is roughly 1,600.

## 3.4

- New complexity measure: `BubbleEntropy`.
- New outcome space: `BubbleSortSwaps`.
- New encoding: `BubbleSortSwapsEncoding`.

## 3.3

- Added the `SequentialPairDistances` outcome space. In the literature, this outcome
    space has been used to compute the "distribution entropy", which can be reproduced
    with `information(Shannon(), SequentialPairDistances(x), x)`. It can of course also
    be used in combination with any other information measure.
- Added the `PairDistanceEncoding` encoding.
- Added the `entropy_distribution` convenience function.

## 3.2

- `missing_outcomes` only works with count-based outcome spaces, which is what it should be doing based on its conceptual definition. Previous signature has been deprecated.
- New function `missing_probabilities` that works with probability estimators and does the same as `missing_outcomes`.

## 3.1

- Pretty printing for `Encoding`s, `OutcomeSpace`s, `ProbabilitiesEstimator`s,
    `InformationMeasure`s, `InformationMeasureEstimator`s and `ComplexityEstimator`s.

## 3.0

ComplexityMeasures.jl has undergone major overhaul of the internal design.
Additionally, a large number of exported names have been renamed. Despite the major
version change, this release does not contain strictly breaking changes. Instead,
deprecations have been put in place everywhere.

The main renames and re-thinking of the library design are:

- We have renamed the concept of "entropy" to "information measure", and `entropy` has
    been renamed to  `information`. We consider as "information measures" anything that is
    a functional of probability mass/density functions, and these are estimated using
    `DiscreteInfoEstimator`s or `DifferentialInfoEstimator`s.
-  We realized that types like `ValueBinning`, `OrdinalPatterns` and `Dispersion` don't
    actually represent probabilities estimators, but *outcome spaces*. To convery this
    fact, from 3.0, these types are subtypes of `OutcomeSpace`.
- Subtypes of `ProbabilitiesEstimator`s now represent distinct ways of estimating
    probabilities from counts or pseudo-counts over some `OutcomeSpace`.
    `RelativeAmount` is the simplest (and default) probabilities estimator.
    `BayesianRegularization`, `Shrinkage` and `AddConstant` are some more complex
    probabilities estimators.

The online documentation now comes with a tutorial that nicely summarizes these new
concepts/changes.

### New library features

- New dedicated counting interface for mapping observations into outcome counts. See
    the `counts_and_outcomes` function and `Counts` struct.
- New function `allprobabilities` that is like `probabilities` but also includes 0
    entries for possible outcomes that were not present in the data.
- New _extropy_ definitions that count as information measures (and thus can be given to
    `information`): `ShannonExtropy`, `RenyiExtropy`, `TsallisExtropy`.
- `StatisticalComplexity` is now compatible with any normalizable `InformationMeasure`
    (previously `EntropyDefinition`).
- `StatisticalComplexity` can now estimate probabilities using any combination of
    `ProbabilitiesEstimator` and `OutcomeSpace`.
- Add the 1976 Lempel-Ziv complexity measure (`LempelZiv76`).
- New entropy definition: identification entropy (`Identification`).
- Minor documentation fixes.
- `GaussianCDFEncoding` now can be used with vector-valued inputs.
- New `LeonenkoProzantoSavani` differential entropy estimator. Works with `Shannon`,
    `Renyi` and `Tsallis` entropies.
- New encodings available: `RelativeMeanEncoding`, `RelativeFirstDifferenceEncoding`,
    `UniqueElementsEncoding` and `CombinationEncoding` (the latter combines multiple
    encodings).
- New `codify` function that encodes sequences of observations (vectors or state space
    sets) into discrete symbol sequences.

### Renaming (deprecated)

- `SymbolicPermutation` is now `OrdinalPatterns`.
- `SymbolicWeightedPermutation` is now `WeightedOrdinalPatterns`.
- `SymbolicAmplitudeAwarePermutation` is now `AmplitudeAwareOrdinalPatterns`.
- `SpatialSymbolicPermutation` is now `SpatialOrdinalPatterns`.

### Other deprecations

- Passing `m` as a positional or keyword argument to ordinal pattern outcome space or
    encoding is deprecated. It is given as a type parameter now, e.g.,
    `OrdinalPatterns{m}(...)` instead of `OrdinalPatterns(m = ..., ...)`.

### Bug fixes

- `outcome_space` for `Dispersion` now correctly returns the all possible **sorted**
    outcomes (as promised by the `outcome_space` docstring).
- `decode` with `GaussianCDFEncoding` now correctly returns only the left-sides of the
    `[0, 1]` subintervals, and always returns the decoded symbol as a `Vector{SVector}`
    (consistent with `RectangularBinEncoding`), regardless of whether the input is a scalar
    or a vector.
- Using the `TransferOperator` outcome space with a `RectangularBinning` or
    `FixedRectangularBinning` with `precise == false` will now trigger a warning.
    This was previously causing random bugs because some bins were encoded as `-1`,
    indicating that the point is outside the binning - even if it wasn't.
- `WaveletOverlap` now computes probabilities (relative energies) over the correct number
    of transform levels. Previously, the *scaling *coefficients for the max transform
    level were incorrectly included, as an extra set of coefficients in addition to the
    (correctly included) wavelet coefficients. This caused a lot of energy to be
    concentrated at low frequencies, even for high-frequency signals. Thus the
    corresponding `Probabilities` had an extra element which in many cases dominated the
    rest of the distribution.

## 2.7.1

- Fix bug in calculation of statistical complexity

## 2.7

- Add generalized statistical complexity as complexity measure.

## 2.6

- Fixed differential entropy "unit" bug caused by erroneous conversion between logarithm
    bases and introduced the `convert_logunit` function to convert between entropies
    computed with different logarithm bases.

## 2.5

- Moved to StateSpaceSets.jl v1 (only renames of `Dataset` to `StateSpaceSet`).

## 2.4

- Rectangular binnings have been reformed to operate based on ranges. This leads to much more intuitive bin sizes and edges. For `RectangularBinning` nothing changes, while for `FixedRectangularBinning` the ranges should be given explicitly. Backwards compatible deprecations have been added.
- This also allows for a new `precise` option that utilizes Base Julia `TwinPrecision` to make more accurate mapping of points to bins at the cost of performance.

## 2.3

- Like differential entropies, discrete entropies now also have their own estimator type.
- The approach of giving both an entropy definition, and an entropy estimator to `entropy` has been dropped. Now the entropy estimators know what definitions they are applied for. This change is a deprecation, i.e., backwards compatible.
- Added `PlugInEntropy` discrete entropy estimator.

## 2.2

- Corrected documentation for `SymbolicPermutation`, `SymbolicAmplitudeAwarePermutation`,
    and `SymbolicWeightedPermutation`, indicating that the outcome space is the set of
    `factorial(m)` *permutations* of the integers `1:m`, not the rank orderings,
    as was stated before.

## 2.1

- Added `Gao` estimator for differential Shannon entropy.
- Added `Lord` estimator for differential Shannon entropy.
- `Probabilities` now wraps `AbstractArray{T, N}` instead of `AbstractVector{T}`, so that it can also represent multidimensional probability mass functions. For vectors, it behaves as before.

## 2.0

The API for Entropies.jl has been completely overhauled, and the package has been renamed to ComplexityMeasures.jl.
Along with the overhaul comes a massive amount of new features, an entirely new API, extendable and educative code, dedicated documentation pages, and more!

We believe it is best to learn all of this by visiting the online documentation.

We tried our best to keep pre-2.0 functions working and throw deprecation warnings.
If we missed code that should be working, let us know by opening an issue.

### Major changes

- Common generic interface function `entropy`, `entropy_normalized` and `maximum` (maximum entropy) that dispatches on different definitions of entropies (e.g `Renyi()` `Shannon()`, `Tsallis()`) and estimated probabilities.
- Convenience functions for common entropies, such as permutation entropy and dispersion entropy still exist.
- New interface `DifferentialEntropyEstimator` that is also used in `entropy`.
- The `base` of the entropy is now a field of the `InformationMeasure` type, not the estimator.
- An entirely new section of entropy-like complexity measures, such as the reverse dispersion entropy.
- Many new estimators, such as `SpatialPermutation` and `PowerSpectrum`.
- Check the online documentation for a comprehensive overview of the changes.

### Minor changes

- No more deprecation warnings for using the old keyword `α` for Renyi entropy.
- The `KozachenkoLeonenko` estimator now correctly fixes its neighbor search to the
    *closest* neighbor only, and its constructor does no longer accept `k` as an input. It also uses correct scaling factor and adapts to dimension.
- Using a logarithm `base` different from `MathConstants.e` now yields correct results
    for `Kraskov` and `KozachenkoLeonenko`.

## main

- New probability estimator `SpatialSymbolicPermutation` suitable for computing spatial permutation entropies
- Introduce Tsallis entropy.

## 1.2

- Added dispersion entropy.

## 1.1

- Introduce convenience function `permentropy`.
- Several type instabilities fixed.

## 1.0

No actual changes, just first major version release.

## 0.12

- Nearest neighbor searches now use Neighborhood.jl and the Theiler window properly.

## 0.11.1

- `probabilities(data, n::Int)` now uses a rectangular binning of `n` bins for each dimension. Before, while not documented as possible in the public API, using integer `n` would take it as the bin size.
