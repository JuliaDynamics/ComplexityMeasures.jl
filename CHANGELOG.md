# CHANGELOG
Changelog is kept with respect to version 0.11 of Entropies.jl.

## 1.1
* Introduce convenience function `permentropy`.
* Several type instabilities fixed.

## 1.0
No actual changes, just first major version release.

## 0.12
* Nearest neighbor searches now use Neighborhood.jl and the Theiler window properly.

## 0.11.1
* `probabilities(data, n::Int)` now uses a rectangular binning of `n` bins for each dimension. Before, while not documented as possible in the public API, using integer `n` would take it as the bin size.
