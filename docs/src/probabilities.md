# Probabilities

```@docs
Probabilities
probabilities
probabilities!
ProbabilitiesEstimator
```

## [Estimators](@id estimators)

### Count occurrences (counting)

```@docs
CountOccurrences
```

### Permutation (symbolic)

```@docs
SymbolicPermutation
```

#### Example

This example reproduces an example from Bandt and Pompe (2002), where the permutation
entropy is compared with the largest Lyapunov exponents from time series of the chaotic
logistic map. Entropy estimates using [`SymbolicWeightedPermutation`](@ref)
and [`SymbolicAmplitudeAwarePermutation`](@ref) are added here for comparison.

```@example MAIN
using DynamicalSystems, CairoMakie

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
    hperm = Entropies.entropy_renyi(x, SymbolicPermutation(m = m, τ = τ), base = base)
    hwtperm = Entropies.entropy_renyi(x, SymbolicWeightedPermutation(m = m, τ = τ), base = base)
    hampperm = Entropies.entropy_renyi(x, SymbolicAmplitudeAwarePermutation(m = m, τ = τ), base = base)

    push!(hs_perm, hperm); push!(hs_wtperm, hwtperm); push!(hs_ampperm, hampperm)
end

fig = Figure()
a1 = Axis(fig[1,1]; ylabel = L"\lambda")
lines!(a1, rs, lyaps); ylims!(a1, (-2, log(2)))
a2 = Axis(fig[2,1]; ylabel = L"h_6 (SP)")
lines!(a2, rs, hs_perm; color = Cycled(2))
a3 = Axis(fig[3,1]; ylabel = L"h_6 (WT)")
lines!(a3, rs, hs_wtperm; color = Cycled(3))
a4 = Axis(fig[4,1]; ylabel = L"h_6 (SAAP)")
lines!(a4, rs, hs_ampperm; color = Cycled(4))
a4.xlabel = L"r"

for a in (a1,a2,a3)
    hidexdecorations!(a, grid = false)
end
fig
```

### Visitation frequency (binning)

```@docs
VisitationFrequency
```

#### Specifying binning/boxes

```@docs
RectangularBinning
```

### Transfer operator (binning)

```@docs
TransferOperator
```

#### Utility methods/types

```@docs
InvariantMeasure
invariantmeasure
transfermatrix
```

### Kernel density

```@docs
NaiveKernel
```

#### Example

Here, we draw some random points from a 2D normal distribution. Then, we use kernel 
density estimation to associate a probability to each point `p`, measured by how many 
points are within radius `1.5` of `p`. Plotting the actual points, along with their 
associated probabilities estimated by the KDE procedure, we get the following surface 
plot.

```@example MAIN
using DynamicalSystems, CairoMakie, Distributions
𝒩 = MvNormal([1, -4], 2)
N = 500
D = Dataset(sort([rand(𝒩) for i = 1:N]))
x, y = columns(D)
p = probabilities(D, NaiveKernel(1.5))
fig, ax = surface(x, y, p.p; axis=(type=Axis3,))
ax.zlabel = "P"
ax.zticklabelsvisible = false
fig
```

### Wavelet

```@docs
WaveletOverlap
```

#### Example

The scale-resolved wavelet entropy should be lower for very regular signals (most of the
energy is contained at one scale) and higher for very irregular signals (energy spread
more out across scales).

```@example MAIN
using DynamicalSystems, CairoMakie
N, a = 1000, 10
t = LinRange(0, 2*a*π, N)

x = sin.(t);
y = sin.(t .+ cos.(t/0.5));
z = sin.(rand(1:15, N) ./ rand(1:10, N))

est = WaveletOverlap()
h_x = entropy_renyi(x, est)
h_y = entropy_renyi(y, est)
h_z = entropy_renyi(z, est)

fig = Figure()
ax = Axis(fig[1,1]; ylabel = "x")
lines!(ax, t, x; color = Cycled(1), label = "h=$(h=round(h_x, sigdigits = 5))");
ay = Axis(fig[2,1]; ylabel = "y")
lines!(ay, t, y; color = Cycled(2), label = "h=$(h=round(h_y, sigdigits = 5))");
az = Axis(fig[3,1]; ylabel = "z", xlabel = "time")
lines!(az, t, z; color = Cycled(3), label = "h=$(h=round(h_z, sigdigits = 5))");
for a in (ax, ay, az); axislegend(a); end
for a in (ax, ay); hidexdecorations!(a; grid=false); end
fig
```