# Examples

## Nearest neighbor direct entropy example

This example reproduces Figure in Charzyńska & Gambin (2016)[^Charzyńska2016]. Both
estimators nicely converge to the "true" entropy with increasing time series length.
For a uniform 1D distribution ``U(0, 1)``, the true entropy is `0`.

```@example MAIN
using DynamicalSystems, CairoMakie, Statistics
using Distributions: Uniform, Normal

Ns = [100:100:500; 1000:1000:10000]
Ekl = Vector{Vector{Float64}}(undef, 0)
Ekr = Vector{Vector{Float64}}(undef, 0)

nreps = 50
for N in Ns
    kl = Float64[]
    kr = Float64[]
    for i = 1:nreps
        pts = Dataset([rand(Uniform(0, 1), 1) for i = 1:N]);

        push!(kl, entropy(KozachenkoLeonenko(w = 0, k = 1, base = MathConstants.e), pts))
        # with k = 1, Kraskov is virtually identical to
        # Kozachenko-Leonenko, so pick a higher number of neighbors
        push!(kr, entropy(Kraskov(w = 0, k = 3, base = MathConstants.e), pts))
    end
    push!(Ekl, kl)
    push!(Ekr, kr)
end

fig = Figure()
ax = Axis(fig[1,1]; ylabel = "entropy (nats)", title = "Kozachenko-Leonenko")
lines!(ax, Ns, mean.(Ekl); color = Cycled(1))
band!(ax, Ns, mean.(Ekl) .+ std.(Ekl), mean.(Ekl) .- std.(Ekl);
color = (Main.COLORS[1], 0.5))

ay = Axis(fig[2,1]; xlabel = "time step", ylabel = "entropy (nats)", title = "Kraskov")
lines!(ay, Ns, mean.(Ekr); color = Cycled(2))
band!(ay, Ns, mean.(Ekr) .+ std.(Ekr), mean.(Ekr) .- std.(Ekr);
color = (Main.COLORS[2], 0.5))

fig
```

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy estimator with applications in systems biology. Entropy, 18(1), 13.

## Permutation entropy example

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
using DynamicalSystems, CairoMakie, Distributions
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
using DynamicalSystems, CairoMakie
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

### Stretched exponential entropy

Here, we reproduce the example from Anteneodo & Plastino (1999)[^Anteneodo1999], showing
how the stretched exponential entropy changes as function of the parameter `η` for a range
of two-element probability distributions given by
`Probabilities([p, 1 - p] for p in 1:0.0:0.01:1.0)`.

```@example stretched_exponential_example
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

```@example
using Entropies, DynamicalSystems, Random, CairoMakie, Distributions

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
est_de = Dispersion(symbolization = GaussianSymbolization(c), m = m, τ = 1)

for (i, window) in enumerate(windows)
    rdes[i] = reverse_dispersion(y[window], est_de; normalize = true)
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
using DynamicalSystems
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