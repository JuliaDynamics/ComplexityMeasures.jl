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

### Curado entropy

Here, we reproduce Figure 2 from Curado & Nobre (2004)[^Curado2004], showing
how the [Curado](@ref) entropy changes as function of the parameter `a` for a range of two-element probability distributions given by
`Probabilities([p, 1 - p] for p in 1:0.0:0.01:1.0)`.

```@example stretched_exponential_example
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
est_rd = ReverseDispersion(symbolization = GaussianSymbolization(c), m = m, τ = 1)
est_de = Dispersion(symbolization = GaussianSymbolization(c), m = m, τ = 1)

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

## Missing dispersion patterns

```@example
using CairoMakie
using DynamicalSystemsBase
using Entropies
using TimeseriesSurrogates
using Statistics

d = Dispersion(m = 3, symbolization = GaussianSymbolization(c = 7))
est = MissingDispersionPatterns(d)
sys = Systems.logistic(0.6; r = 4.0)
normalize = true
Ls = collect(100:100:1000)
nL = length(Ls)
nreps = 50
method = WLS(IAAFT(), true)

r_det, r_noise = zeros(length(Ls)), zeros(length(Ls))
r_det_surr, r_noise_surr = [zeros(nreps) for L in Ls], [zeros(nreps) for L in Ls]
y = rand(maximum(Ls))

for (i, L) in enumerate(Ls)
    # Deterministic time series
    x = trajectory(sys, L - 1, Ttr = 5000)
    sx = surrogenerator(x, method)
    r_det[i] = complexity_normalized(est, x)
    r_det_surr[i][:] = [complexity_normalized(est, sx()) for j = 1:nreps]
   
    # Random time series
    r_noise[i] = complexity_normalized(est, y[1:L])
    sy = surrogenerator(y[1:L], method)
    r_noise_surr[i][:] = [complexity_normalized(est, sy()) for j = 1:nreps]
end

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = "Time series length (L)", 
    ylabel = "# missing dispersion patterns (normalized)"
)

lines!(ax, Ls, r_det, label = "logistic(x0 = 0.6; r = 4.0)", color = :black)
lines!(ax, Ls, r_noise, label = "Uniform noise", color = :red)
for i = 1:nL
    if i == 1
        boxplot!(ax, fill(Ls[i], nL), r_det_surr[i]; width = 50, color = :black, 
            label = "WIAAFT surrogates (logistic)")
         boxplot!(ax, fill(Ls[i], nL), r_noise_surr[i]; width = 50, color = :red, 
            label = "WIAAFT surrogates (noise)")
    else
        boxplot!(ax, fill(Ls[i], nL), r_det_surr[i]; width = 50, color = :black)
        boxplot!(ax, fill(Ls[i], nL), r_noise_surr[i]; width = 50, color = :red)
    end
end
axislegend(position = :rc)
ylims!(0, 1.1)

fig
```

We don't need to actually to compute the quantiles here to see that for the logistic
map, across all time series lengths, the ``N_{MDP}`` values are above the extremal values 
of the ``N_{MDP}`` values for the surrogate ensembles. Thus, we
conclude that the logistic map time series has nonlinearity (well, of course).

For the univariate noise time series, there is considerable overlap between ``N_{MDP}``
for the surrogate distributions and the original signal, so we can't claim nonlinearity
for this signal.

Of course, to robustly reject the null hypothesis, we'd need to generate a sufficient number
of surrogate realizations, and actually compute quantiles to compare with.

[^Zhou2022]: Zhou, Q., Shang, P., & Zhang, B. (2022). Using missing dispersion patterns
    to detect determinism and nonlinearity in time series data. Nonlinear Dynamics, 1-20.

## Approximate entropy

Here, we reproduce the Henon map example with ``R=0.8`` from Pincus (1991),
comparing our values with relevant values from table 1 in Pincus (1991).

We use `DiscreteDynamicalSystem` from `DynamicalSystems.jl` to represent the map,
and use the `trajectory` function from the same package to iterate the map
for different initial conditions, for multiple time series lengths.

Finally, we summarize our results in box plots and compare the values to those
obtained by Pincus (1991).

```@example
using Entropies, DynamicalSystems, CairoMakie

# Equation 13 in Pincus (1991)
function eom_henon(u, p, n)
    R = p[1]
    x, y = u
    dx = R*y + 1 - 1.4*x^2
    dy = 0.3*R*x

    return SVector{2}(dx, dy)
end

function henon(; u₀ = rand(2), R = 0.8)
    DiscreteDynamicalSystem(eom_henon, u₀, [R])
end

ts_lengths = [300, 1000, 2000, 3000]
nreps = 100
apens_08 = [zeros(nreps) for i = 1:length(ts_lengths)]

# For some initial conditions, the Henon map as specified here blows up,
# so we need to check for infinite values.
containsinf(x) = any(isinf.(x))

c = ApproximateEntropy(r = 0.05, m = 2)

for (i, L) in enumerate(ts_lengths)
    k = 1
    while k <= nreps
        sys = henon(u₀ = rand(2), R = 0.8)
        t = trajectory(sys, L, Ttr = 5000)

        if !any([containsinf(tᵢ) for tᵢ in t])
            x, y = columns(t)
            apens_08[i][k] = complexity(c, x)
            k += 1
        end
    end
end

fig = Figure()

# Example time series
a1 = Axis(fig[1,1]; xlabel = "Time (t)", ylabel = "Value")
sys = henon(u₀ = [0.5, 0.1], R = 0.8)
x, y = columns(trajectory(sys, 100, Ttr = 500))
lines!(a1, 1:length(x), x, label = "x")
lines!(a1, 1:length(y), y, label = "y")

# Approximate entropy values, compared to those of the original paper (black dots).
a2 = Axis(fig[2, 1]; 
    xlabel = "Time series length (L)", 
    ylabel = "ApEn(m = 2, r = 0.05)")

# hacky boxplot, but this seems to be how it's done in Makie at the moment
n = length(ts_lengths)
for i = 1:n
    boxplot!(a2, fill(ts_lengths[i], n), apens_08[i];
        width = 200)
end

scatter!(a2, ts_lengths, [0.337, 0.385, NaN, 0.394]; 
    label = "Pincus (1991)", color = :black)
fig
```

## Sample entropy

Completely regular signals should have sample entropy approaching zero, while
less regular signals should have higher sample entropy.

```@example
using DynamicalSystems, CairoMakie
N, a = 2000, 10
t = LinRange(0, 2*a*π, N)

x = repeat([-5:5 |> collect; 4:-1:-4 |> collect], N ÷ 20);
y = sin.(t .+ cos.(t/0.5));
z = rand(N)

h_x, h_y, h_z = map(t -> complexity(SampleEntropy(t), t), (x, y, z))

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

Next, we compare the sample entropy obtained for different values of the radius `r` for
uniform noise, normally distributed noise, and a periodic signal.

```@example
using Entropies, CairoMakie, Distributions
N = 2000
x_U = rand(N)
x_N = rand(Normal(0, 3), N)
x_periodic = repeat(rand(20), N ÷ 20)

x_U .= (x_U .- mean(x_U)) ./ std(x_U)
x_N .= (x_N .- mean(x_N)) ./ std(x_N)
x_periodic .= (x_periodic .- mean(x_periodic)) ./ std(x_periodic)

rs = 10 .^ range(-1, 0, length = 30)
base = 2
m = 2
hs_U = [complexity_normalized(SampleEntropy(m = m, r = r), x_U) for r in rs]
hs_N = [complexity_normalized(SampleEntropy(m = m, r = r), x_N) for r in rs]
hs_periodic = [complexity_normalized(SampleEntropy(m = m, r = r), x_periodic) for r in rs]

fig = Figure()
# Time series
a1 = Axis(fig[1,1]; xlabel = "r", ylabel = "Sample entropy")
lines!(a1, rs, hs_U, label = "Uniform noise, U(0, 1)")
lines!(a1, rs, hs_N, label = "Gaussian noise, N(0, 1)")
lines!(a1, rs, hs_periodic, label = "Periodic signal")
axislegend()
fig
```
