# Permutation (symbolic)

```@docs
SymbolicPermutation
```

## Example

This example reproduces an example from Bandt and Pompe (2002), where the permutation
entropy is compared with the largest Lyapunov exponents from time series of the chaotic 
logistic map. Entropy estimates using [`SymbolicWeightedPermutation`](@ref)
and [`SymbolicAmplitudeAwarePermutation`](@ref) are added here for comparison.

```@example
using DynamicalSystems, PyPlot, Entropies

ds = Systems.logistic()
rs = 3.4:0.001:4
N_lyap, N_ent = 100000, 10000
m, τ = 6, 1 # Symbol size/dimension and embedding lag

# Generate one time series for each value of the logistic parameter r
lyaps = Float64[]
hs_perm = Float64[]
hs_wtperm = Float64[]
hs_ampperm = Float64[]

base = Base.MathConstants.e
for r in rs
    ds.p[1] = r
    push!(lyaps, lyapunov(ds, N_lyap))

    x = trajectory(ds, N_ent) # time series
    hperm = Entropies.genentropy(x, SymbolicPermutation(m = m, τ = τ), base = base)
    hwtperm = Entropies.genentropy(x, SymbolicWeightedPermutation(m = m, τ = τ), base = base)
    hampperm = Entropies.genentropy(x, SymbolicAmplitudeAwarePermutation(m = m, τ = τ), base = base)

    push!(hs_perm, hperm); push!(hs_wtperm, hwtperm); push!(hs_ampperm, hampperm)
end

f = figure(figsize = (6, 8))
a1 = subplot(411)
plot(rs, lyaps); ylim(-2, log(2)); ylabel("\$\\lambda\$")
a1.axes.get_xaxis().set_ticklabels([])
xlim(rs[1], rs[end]);

a2 = subplot(412)
plot(rs, hs_perm; color = "C2"); xlim(rs[1], rs[end]);
xlabel(""); ylabel("\$h_6 (SP)\$")

a3 = subplot(413)
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
