
"""
    WalkthroughEntropy

The walkthrough entropy method (Stoop et al., 2021)[^Stoop2021].

Does not work with `genentropy`, but combination with `entropygenerator`, we can use
this estimator to compute walkthrough entropy for multiple `n` with a single initialization
step (instead of initializing once per `n`).

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = "abc"^2
"abcabc"

julia> wg = entropygenerator(x, WalkthroughEntropy());

julia> [wg(n) for n = 1:length(x)]
6-element Vector{Float64}:
  1.0986122886681098
  1.3217558399823195
  0.9162907318741551
  1.3217558399823195
  1.0986122886681098
 -0.0
```

See also: [`entropygenerator`](@ref).

[^Stoop2021]: Stoop, R. L., Stoop, N., Kanders, K., & Stoop, R. (2021). Excess entropies suggest the physiology of neurons to be primed for higher-level computation. Physical Review Letters, 127(14), 148101.
"""
struct WalkthroughEntropy <: EntropyEstimator end

include("walkthrough_prob.jl")
include("walkthrough_entropy.jl")
