# Entropies.jl Examples

## Indirect entropy (nearest neighbors)

Here, we reproduce Figure 1 in Charzyńska & Gambin (2016)[^Charzyńska2016]. Their example
demonstrates how the [`Kraskov`](@ref) and [`KozachenkoLeonenko`](@ref) nearest neighbor
based estimators converge towards the true entropy value for increasing time series length.
We extend their example with [`Zhu`](@ref) and [`ZhuSingh`](@ref) estimators, which are also
based on nearest neighbor searches.

Input data are from a uniform 1D distribution ``U(0, 1)``, for which the true entropy is
`ln(1 - 0) = 0`).

```@example MAIN
using Entropies
using DynamicalSystemsBase, CairoMakie, Statistics
using Distributions: Uniform, Normal

# Define estimators
base = MathConstants.e # shouldn't really matter here, because the target entropy is 0.
w = 0 # Theiler window of 0 (only exclude the point itself during neighbor searches)
estimators = [
    # with k = 1, Kraskov is virtually identical to
    # Kozachenko-Leonenko, so pick a higher number of neighbors for Kraskov
    Kraskov(; k = 3, w, base),
    KozachenkoLeonenko(; w, base),
    Zhu(; k = 3, w, base),
    ZhuSingh(; k = 3, w, base),
]
labels = ["KozachenkoLeonenko", "Kraskov", "Zhu", "ZhuSingh"]

# Test each estimator `nreps` times over time series of varying length.
nreps = 50
Ns = [100:100:500; 1000:1000:10000]

Hs_uniform = [[zeros(nreps) for N in Ns] for e in estimators]
for (i, e) in enumerate(estimators)
    for j = 1:nreps
        pts = rand(Uniform(0, 1), maximum(Ns)) |> Dataset
        for (k, N) in enumerate(Ns)
            Hs_uniform[i][k][j] = entropy(e, pts[1:N])
        end
    end
end

fig = Figure(resolution = (600, length(estimators) * 200))
for (i, e) in enumerate(estimators)
    Hs = Hs_uniform[i]
    ax = Axis(fig[i,1]; ylabel = "h (nats)")
    lines!(ax, Ns, mean.(Hs); color = Cycled(i), label = labels[i])
    band!(ax, Ns, mean.(Hs) .+ std.(Hs), mean.(Hs) .- std.(Hs);
    color = (Main.COLORS[i], 0.5))
    ylims!(-0.25, 0.25)
    axislegend()
end

fig
```

## Indirect entropy (order statistics)

Entropies.jl also provides entropy estimators based on
[order statistics](https://en.wikipedia.org/wiki/Order_statistic). These estimators
are only defined for scalar-valued vectors, so we pass the data as `Vector{<:Real}`s instead
of `Dataset`s, as we did for the nearest-neighbor estimators above.

Here, we show how the [`Vasicek`](@ref), [`Ebrahimi`](@ref), [`AlizadehArghami`](@ref) 
and [`Correa`](@ref) direct [`Shannon`](@ref) entropy estimators, with increasing sample size,
approach zero for samples from a uniform distribution on  `[0, 1]`. The true entropy value in
nats for this distribution is `ln(1 - 0) = 0`.

```@example MAIN
using Entropies
using Statistics
using Distributions
using CairoMakie

# Define estimators
base = MathConstants.e # shouldn't really matter here, because the target entropy is 0.
# just provide types here, they are instantiated inside the loop
estimators = [Vasicek, Ebrahimi, AlizadehArghami, Correa]
labels = ["Vasicek", "Ebrahimi", "AlizadehArghami", "Correa"]

# Test each estimator `nreps` times over time series of varying length.
Ns = [100:100:500; 1000:1000:10000]
nreps = 30

Hs_uniform = [[zeros(nreps) for N in Ns] for e in estimators]
for (i, e) in enumerate(estimators)
    for j = 1:nreps
        pts = rand(Uniform(0, 1), maximum(Ns)) # raw timeseries, not a `Dataset`
        for (k, N) in enumerate(Ns)
            m = floor(Int, N / 100) # Scale `m` to timeseries length
            est = e(; m, base) # Instantiate estimator with current `m`
            Hs_uniform[i][k][j] = entropy(est, pts[1:N])
        end
    end
end

fig = Figure(resolution = (600, length(estimators) * 200))
for (i, e) in enumerate(estimators)
    Hs = Hs_uniform[i]
    ax = Axis(fig[i,1]; ylabel = "h (nats)")
    lines!(ax, Ns, mean.(Hs); color = Cycled(i), label = labels[i])
    band!(ax, Ns, mean.(Hs) .+ std.(Hs), mean.(Hs) .- std.(Hs);
    color = (Main.COLORS[i], 0.5))
    ylims!(-0.25, 0.25)
    axislegend()
end

fig
```

As for the nearest neighbor estimators, both estimators also approach the
true entropy value for this example, but is negatively biased for small sample sizes.

## Permutation entropy example

This example reproduces an example from Bandt and Pompe (2002), where the permutation
entropy is compared with the largest Lyapunov exponents from time series of the chaotic
logistic map. Entropy estimates using [`SymbolicWeightedPermutation`](@ref)
and [`SymbolicAmplitudeAwarePermutation`](@ref) are added here for comparison.

```@example MAIN
using DynamicalSystemsBase, CairoMakie, ChaosTools

ds = Systems.logistic()
rs = 3.4:0.001:4
N_lyap, N_ent = 100000, 10000
m, τ = 6, 1 # Symbol size/dimension and embedding lag

# Generate one time series for each value of the logistic parameter r
lyaps, hs_perm, hs_wtperm, hs_ampperm = [zeros(length(rs)) for _ in 1:4]

for (i, r) in enumerate(rs)
    ds.p[1] = r
    lyaps[i] = lyapunov(ds, N_lyap)

    x = trajectory(ds, N_ent) # time series
    hperm = entropy(x, SymbolicPermutation(; m, τ))
    hwtperm = entropy(x, SymbolicWeightedPermutation(; m, τ))
    hampperm = entropy(x, SymbolicAmplitudeAwarePermutation(; m, τ))

    hs_perm[i] = hperm; hs_wtperm[i] = hwtperm; hs_ampperm[i] = hampperm
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

## Kernel density example

Here, we draw some random points from a 2D normal distribution. Then, we use kernel density estimation to associate a probability to each point `p`, measured by how many points are within radius `1.5` of `p`. Plotting the actual points, along with their associated probabilities estimated by the KDE procedure, we get the following surface plot.

```@example MAIN
using Entropies
using DelayEmbeddings
using DynamicalSystemsBase, CairoMakie, Distributions
𝒩 = MvNormal([1, -4], 2)
N = 500
D = Dataset(sort([rand(𝒩) for i = 1:N]))
x, y = columns(D)
p = probabilities(D, NaiveKernel(1.5))
fig, ax = scatter(D[:, 1], D[:, 2], zeros(N);
    markersize=8, axis=(type = Axis3,)
)
surface!(ax, x, y, p.p)
ax.zlabel = "P"
ax.zticklabelsvisible = false
fig
```

## Wavelet entropy example

The scale-resolved wavelet entropy should be lower for very regular signals (most of the
energy is contained at one scale) and higher for very irregular signals (energy spread
more out across scales).

```@example MAIN
using DynamicalSystemsBase, CairoMakie
N, a = 1000, 10
t = LinRange(0, 2*a*π, N)

x = sin.(t);
y = sin.(t .+ cos.(t/0.5));
z = sin.(rand(1:15, N) ./ rand(1:10, N))

h_x = entropy_wavelet(x)
h_y = entropy_wavelet(y)
h_z = entropy_wavelet(z)

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

## Properties of different entropies

Here, we show the sensitivity of the various entropies to variations in their parameters.

### Curado entropy

Here, we reproduce Figure 2 from Curado & Nobre (2004)[^Curado2004], showing
how the [Curado](@ref) entropy changes as function of the parameter `a` for a range of two-element probability distributions given by
`Probabilities([p, 1 - p] for p in 1:0.0:0.01:1.0)`.

```@example MAIN
using Entropies, CairoMakie
bs = [1.0, 1.5, 2.0, 3.0, 4.0, 10.0]
ps = [Probabilities([p, 1 - p]) for p = 0.0:0.01:1.0]
hs = [[entropy(Curado(; b = b), p) for p in ps] for b in bs]
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "p", ylabel = "H(p)")
pp = [p[1] for p in ps]
for (i, b) in enumerate(bs)
    lines!(ax, pp, hs[i], label = "b=$b", color = Cycled(i))
end
axislegend(ax)
fig
```

[^Curado2004]: Curado, E. M., & Nobre, F. D. (2004). On the stability of analytic
    entropic forms. Physica A: Statistical Mechanics and its Applications, 335(1-2), 94-106.

### Stretched exponential entropy

Here, we reproduce the example from Anteneodo & Plastino (1999)[^Anteneodo1999], showing
how the stretched exponential entropy changes as function of the parameter `η` for a range
of two-element probability distributions given by
`Probabilities([p, 1 - p] for p in 1:0.0:0.01:1.0)`.

```@example MAIN
using Entropies, SpecialFunctions, CairoMakie
ηs = [0.01, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0]
ps = [Probabilities([p, 1 - p]) for p = 0.0:0.01:1.0]

hs_norm = [[entropy(StretchedExponential( η = η), p) / gamma((η + 1)/η) for p in ps] for η in ηs]
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "p", ylabel = "H(p)")
pp = [p[1] for p in ps]

for (i, η) in enumerate(ηs)
    lines!(ax, pp, hs_norm[i], label = "η=$η")
end
axislegend(ax)
fig
```

[^Anteneodo1999]: Anteneodo, C., & Plastino, A. R. (1999). Maximum entropy approach to
    stretched exponential probability distributions. Journal of Physics A: Mathematical
    and General, 32(7), 1089.

## [Dispersion and reverse dispersion entropy](@id dispersion_examples)

Here we reproduce parts of figure 3 in Li et al. (2019), computing reverse and regular dispersion entropy for a time series consisting of normally distributed noise with a single spike in the middle of the signal. We compute the entropies over a range subsets of the data, using a sliding window consisting of 70 data points, stepping the window 10 time steps at a time.

Note: the results here are not exactly the same as in the original paper, because Li et
al. (2019) base their examples on randomly generated numbers and do not provide code that
specify random number seeds.

```@example MAIN
using Entropies, DynamicalSystemsBase, Random, CairoMakie, Distributions

n = 1000
ts = 1:n
x = [i == n ÷ 2 ? 50.0 : 0.0 for i in ts]
rng = Random.default_rng()
s = rand(rng, Normal(0, 1), n)
y = x .+ s

ws = 70
windows = [t:t+ws for t in 1:10:n-ws]
rdes = zeros(length(windows))
des = zeros(length(windows))
pes = zeros(length(windows))

m, c = 2, 6
est_rd = ReverseDispersion(encoding = GaussianMapping(c), m = m, τ = 1)
est_de = Dispersion(encoding = GaussianMapping(c), m = m, τ = 1)

for (i, window) in enumerate(windows)
    rdes[i] = complexity_normalized(est_rd, y[window])
    des[i] = entropy_normalized(Renyi(), y[window], est_de)
end

fig = Figure()

a1 = Axis(fig[1,1]; xlabel = "Time step", ylabel = "Value")
lines!(a1, ts, y)
display(fig)

a2 = Axis(fig[2, 1]; xlabel = "Time step", ylabel = "Value")
p_rde = scatterlines!([first(w) for w in windows], rdes,
    label = "Reverse dispersion entropy",
    color = :black,
    markercolor = :black, marker = '●')
p_de = scatterlines!([first(w) for w in windows], des,
    label = "Dispersion entropy",
    color = :red,
    markercolor = :red, marker = 'x', markersize = 20)

axislegend(position = :rc)
ylims!(0, max(maximum(pes), 1))
fig
```

[^Rostaghi2016]: Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
[^Li2019]: Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: a new
    complexity measure for sensor signal. Sensors, 19(23), 5203.

## Normalized entropy for comparing different signals

When comparing different signals or signals that have different length, it is best to normalize entropies so that the "complexity" or "disorder" quantification is directly comparable between signals. Here is an example based on the [Wavelet entropy example](@ref) (where we use the spectral entropy instead of the wavelet entropy):

```@example MAIN
using DynamicalSystemsBase
N1, N2, a = 101, 100001, 10

for N in (N1, N2)
    local t = LinRange(0, 2*a*π, N)
    local x = sin.(t) # periodic
    local y = sin.(t .+ cos.(t/0.5)) # periodic, complex spectrum
    local z = sin.(rand(1:15, N) ./ rand(1:10, N)) # random
    local w = trajectory(Systems.lorenz(), N÷10; Δt = 0.1, Ttr = 100)[:, 1] # chaotic

    for q in (x, y, z, w)
        h = entropy(q, PowerSpectrum())
        n = entropy_normalized(q, PowerSpectrum())
        println("entropy: $(h), normalized: $(n).")
    end
end
```

You see that while the direct entropy values of the chaotic and noisy signals change massively with `N` but they are almost the same for the normalized version.
For the regular signals, the entropy decreases nevertheless because the noise contribution of the Fourier computation becomes less significant.
