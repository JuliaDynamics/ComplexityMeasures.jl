# Permutation (symbolic)

```@docs
SymbolicPermutation
```

```@docs 
entropy(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
```

```@docs 
probabilities(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
```

## Example 


This example reproduces the permutation entropy example on the logistic map from Bandt and Pompe (2002).

```@example
using DynamicalSystems, PyPlot, Entropies

ds = Systems.logistic()
rs = 3.5:0.001:4
N_lyap, N_ent = 100000, 10000

# Generate one time series for each value of the logistic parameter r
lyaps, hs_entropies, hs_chaostools = Float64[], Float64[], Float64[]

for r in rs
    ds.p[1] = r
    push!(lyaps, lyapunov(ds, N_lyap))
    
    # For 1D systems `trajectory` returns a vector, so embed it using τs
    # to get the correct 6d dimension on the embedding
    x = trajectory(ds, N_ent)
    τs = ([-i for i in 0:6-1]...,) # embedding lags
    emb = genembed(x, τs)
    
    # Pre-allocate symbol vector, one symbol for each point in the embedding - this is faster!
    s = zeros(Int, length(emb));
    push!(hs_entropies, entropy!(s, emb, SymbolicPermutation(6, b = Base.MathConstants.e)))

    # Old ChaosTools.jl style estimation
    push!(hs_chaostools, permentropy(x, 6))
end

f = figure(figsize = (10,6))
a1 = subplot(311)
plot(rs, lyaps); ylim(-2, log(2)); ylabel("\$\\lambda\$")
a1.axes.get_xaxis().set_ticklabels([])
xlim(rs[1], rs[end]);

a2 = subplot(312)
plot(rs, hs_chaostools; color = "C1"); xlim(rs[1], rs[end]); 
xlabel("\$r\$"); ylabel("\$h_6 (ChaosTools.jl)\$")

a3 = subplot(313)
plot(rs, hs_entropies; color = "C2"); xlim(rs[1], rs[end]); 
xlabel("\$r\$"); ylabel("\$h_6 (Entropies.jl)\$")
tight_layout()
savefig("permentropy.png")
```

![](permentropy.png)

## Utils

Some convenience functions for symbolization are provided.

```@docs 
symbolize
encode_motif
```