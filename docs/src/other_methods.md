# Methods

## Shannon entropies

Some methods in the literature compute Shannon entropy in ways that don't explicitly result in probability distributions. Hence, they can't be used with [`probabilities`](@ref), and appear instead as stand-alone functions.

### Nearest neighbors

```@docs
entropy_kraskov
entropy_kozachenkoleonenko
```

#### Example

This example reproduces Figure in Charzyńska & Gambin (2016)[^Charzyńska2016]. Both
estimators nicely converge to the true entropy with increasing time series length.
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
