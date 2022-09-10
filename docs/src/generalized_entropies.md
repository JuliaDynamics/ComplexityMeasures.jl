# Entropies

## Rényi (generalized) entropy

```@docs
Entropies.entropy_renyi
```

## Tsallis (generalized) entropy

```@docs
Entropies.entropy_tsallis
```

## Shannon entropy (convenience)
```@docs
entropy_shannon
```

## Indirect entropies
Here we list functions which compute Shannon entropies via alternate means, without explicitly computing some probability distributions.

### Nearest neighbors entropy
```@docs
entropy_kraskov
entropy_kozachenkoleonenko
```

## Convenience functions
In this subsection we expand documentation strings of "entropy names" that are used commonly in the literature, such as "permutation entropy". As we made clear in [API & terminology](@ref), these are just the existing Shannon/Rényi/Tsallis entropy with a particularly chosen probability estimator.