# CHANGELOG
Changelog is kept with respect to version 0.11 of Entropies.jl.

## 0.11.1
* `probabilities(data, n::Int)` now uses a rectangular binning of `n` bins for each dimension. Before, while not documented as possible in the public API, using integer `n` would take it as the bin size.
