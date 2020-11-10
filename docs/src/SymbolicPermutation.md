# Permutation (symbolic)

```@docs
SymbolicPermutation
```

## Example

This example reproduces the permutation entropy example on the logistic map from Bandt and Pompe (2002).

```@example
using DynamicalSystems, PyPlot, Entropies

ds = Systems.logistic()
rs = 3.4:0.001:4
N_lyap, N_ent = 100000, 10000
m = 6 # Symbol size/dimension

# Generate one time series for each value of the logistic parameter r
lyaps, hs_entropies = Float64[], Float64[]
hs_wtperm = Float64[]
hs_ampperm = Float64[]

for r in rs
    ds.p[1] = r
    push!(lyaps, lyapunov(ds, N_lyap))

    # For 1D systems `trajectory` returns a vector, so embed it using τs
    # to get the correct 6d dimension on the embedding
    x = trajectory(ds, N_ent)
    τs = ([-i for i in 0:m-1]...,) # embedding lags
    emb = genembed(x, τs)

    push!(hs_entropies, Entropies.genentropy(emb, SymbolicPermutation(), base = Base.MathConstants.e))
    push!(hs_wtperm, Entropies.genentropy(emb, SymbolicWeightedPermutation(), base = Base.MathConstants.e))
    push!(hs_ampperm, Entropies.genentropy(emb, SymbolicAmplitudeAwarePermutation(), base = Base.MathConstants.e))
end

f = figure(figsize = (6, 23))
a1 = subplot(411)
plot(rs, lyaps); ylim(-2, log(2)); ylabel("\$\\lambda\$")
a1.axes.get_xaxis().set_ticklabels([])
xlim(rs[1], rs[end]);

a2 = subplot(413)
plot(rs, hs_entropies; color = "C2"); xlim(rs[1], rs[end]);
xlabel(""); ylabel("\$h_6 (SP)\$")

a3 = subplot(414)
plot(rs, hs_wtperm; color = "C3"); xlim(rs[1], rs[end]);
xlabel(""); ylabel("\$h_6 (SWP)\$")

a4 = subplot(414)
plot(rs, hs_ampperm; color = "C4"); xlim(rs[1], rs[end]);
xlabel("\$r\$"); ylabel("\$h_6 (SAAP)\$")
tight_layout()
savefig("permentropy.png")
```

![](permentropy.png)

## Utility methods

Some convenience functions for symbolization are provided.

```@docs
symbolize
encode_motif
```
