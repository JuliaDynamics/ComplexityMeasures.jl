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

        push!(kl, entropy_kozachenkoleonenko(pts, w = 0, k = 1))
        # with k = 1, Kraskov is virtually identical to
        # Kozachenko-Leonenko, so pick a higher number of neighbors
        push!(kr, entropy_kraskov(pts, w = 0, k = 3))
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

## [Dispersion probabilities and entropies](@id dispersion_examples)

### Dispersion entropy

#### Symbolizing using cumulative distribution functions

Assume we have a univariate time series $X = \{x_i\}_{i=1}^N$. The dispersion entropy algorithm (Rostaghi & Azami, 2016)[^Rostaghi2016] first maps each $x_i$ to a new real number $y_i \in [0, 1]$ by using the normal cumulative distribution function (CDF), $x_i \to y_i : y_i = \dfrac{1}{\sigma \sqrt{2\pi}}\int_{-\infty}^{x_i} e^{(-(x_i - \mu)^2)/(2\sigma^2)}$, where $\mu$ and $\sigma$ are the empirical mean and standard deviation of $X$. Other choices of CDFs are also possible, but in Entropies.jl currently only implements the normal CDF ([`GaussianSymbolization`](@ref)), which was used in the original dispersion entropy paper.

Next, each $y_i$ is linearly mapped to an integer $z_i \in [1, 2, \ldots, c]$ using the map $y_i \to z_i : z_i = R(y_i(c-1) + 0.5)$, where $c$ is the number of categories and $R$ indicates rounding up to the nearest integer. This procedure subdivides the interval $[0, 1]$ into a set of subintervals that form a covering of $[0, 1]$, and assigns each $y_i$ to one of these subintervals. The original time series $X$ is thus transformed to a symbol time series $S = \{s_i\}_{i=1}^N$, where $s_i \in [1, 2, \ldots, c]$.

#### Dispersion patterns

In the next step, the symbol time series $S$ is embedded into an $m$-dimensional integer-valued time series, using an embedding lag of $\tau = 1$, which yields a total of $N - (m - 1)\tau$ points. Because each $z_i$ can take on $c$ different values, and each embedding point has $m$ values, there are $c^m$ possible values that each embedding point can take. Each embedding vector is called a "dispersion pattern". Why? Let's consider the case when $m = 5$ and $c = 3$, and use some very imprecise terminology for illustration:

When $c = 3$, the "outliers" below the mean are in one group, values close to the mean are in one group, and "outliers" above the mean are in a third group. Then the embedding vector $[2, 2, 2, 2, 2]$ consists of values that are relatively close together (close to the mean), so it represents a set of numbers that are not very spread out (less dispersed). The embedding vector $[1, 1, 2, 3, 3]$, however, represents numbers that are much more spread out (more dispersed), because the categories representing "outliers" both above and below the mean are represented, not only values close to the mean.

A probability distribution $P = \{p_i \}_{i=1}^{c^m}$, where $\sum_i^{c^m} p_i = 1$, can then be estimated by counting and sum-normalising the distribution of dispersion patterns among the embedding vectors. In Entropies.jl, the entire procedure above is performed by the [`Dispersion`](@ref) probabilities estimator (note: in our implementation, dispersion patterns which are not encountered are not counted, so the probabilities you get are always non-zero). Here's an example:

```@example dispersion_entropy
using Entropies
x = repeat([0.5, 0.7, 0.1, -1.0, 1.11, 2.22, 4.4, 0.2, 0.2, 0.1], 10);
c, m = 3, 5
est = Dispersion(s = GaussianSymbolization(c), m = m)
probs = probabilities(x, est)
```

Dispersion entropy is then computed by feeding these probabilites into the formula for 
generalized Renyi entropy, i.e.

```@example dispersion_entropy
entropy_renyi(probs, base = MathConstants.e)
```

### Reverse dispersion entropy

Li et al. (2021)[^Li2019] defines the reverse dispersion entropy as

```math
H_{rde} = \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2.
```

where the probabilities $p_i$ are obtained precisely as for the dispersion entropy.

The minimum value of $H_{rde}$ is zero and occurs precisely when the probability 
distribution is flat, which occurs when all $p_i$s are equal to $1/c^m$. $H_{rde}$ can 
therefore be said to be a measure of how far the dispersion pattern probability 
distribution is from white noise.

#### A clarification on notation

With ambiguous notation, Li et al. claim that

$H_{rde} = \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2 = \sum_{i = 1}^{c^m} p_i^2 - \frac{1}{c^m}.$

But on the right-hand side of the equality, does the constant term appear within or outside the sum?

Let's see. Using (in step 4) that $P$ is a probability distribution by construction, we see that the constant must appear *outside* the sum:

```math
\begin{align}
H_{rde} &= \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2 
= \sum_{i=1}^{c^m} p_i^2 - \frac{2p_i}{c^m} + \frac{1}{c^{2m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \left(\sum_i^{c^m} \frac{2p_i}{c^m}\right) + \left( \sum_{i=1}^{c^m} \dfrac{1}{{c^{2m}}} \right) \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \left(\frac{2}{c^m} \sum_{i=1}^{c^m} p_i \right) +  \dfrac{c^m}{c^{2m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \frac{2}{c^m} (1) +  \dfrac{1}{c^{m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \dfrac{1}{c^{m}}. \\
\end{align}
```

### Example: dispersion entropy vs reverse dispersion entropy

Here we reproduce parts of figure 3 in Li et al. (2019), computing reverse and regular dispersion entropy for a time series consisting of normally distributed noise with a single spike in the middle of the signal. We compute the entropies over a range subsets of the data, using a sliding window consisting of 70 data points, stepping the window 10 time stepts at a time.

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
scheme = GaussianSymbolization(c)
est_de = Dispersion(s = scheme, m = m, τ = 1, normalize = true)

for (i, window) in enumerate(windows)
    rdes[i] = reverse_dispersion(y[window];
        s = scheme, m = m, τ = 1, normalize = true)
    des[i] = entropy_renyi(y[window], est_de)
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