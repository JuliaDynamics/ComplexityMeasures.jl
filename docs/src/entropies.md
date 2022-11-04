# Entropies

```@docs
entropy
entropy!
```

## Rényi (generalized) entropy

```@docs
Renyi
```

## Tsallis (generalized) entropy

```@docs
Tsallis
```

## Shannon entropy (convenience)

```@docs
Shannon
```

## Curado entropy

```@docs
Curado
```

## Stretched exponental entropy

```@docs
StretchedExponential
```

## Normalized entropies

```@docs
entropy_maximum
entropy_normalized
```

## Indirect entropies

Here we list functions which compute Shannon entropies via alternate means, without explicitly computing some probability distributions and then using the Shannon formula.

### Nearest neighbors entropy

```@docs
Kraskov
KozachenkoLeonenko
Zhu
ZhuSingh
```

## Convenience functions

In this subsection we expand documentation strings of "entropy names" that are used commonly in the literature, such as "permutation entropy". As we made clear in [API & terminology](@ref), these are just the existing Shannon entropy with a particularly chosen probability estimator. We have only defined convenience functions for the most used names, and arbitrary more specialized convenience functions can be easily defined in a couple lines of code.

```@docs
entropy_permutation
entropy_spatial_permutation
entropy_wavelet
entropy_dispersion
```
