# Approximate entropy

```@docs
ApproximateEntropy
approx_entropy
```

## Example

Here, we reproduce the Henon map example with ``R=0.8`` from Pincus (1991),
comparing our values with relevant values from table 1 in Pincus (1991).

We use `DiscreteDynamicalSystem` from `DynamicalSystems.jl` to represent the map,
and use the `trajectory` function from the same package to iterate the map
for different initial conditions, for multiple time series lengths.

Finally, we summarize our results in box plots and compare the values to those
obtained by Pincus (1991).

```@example
using DynamicalSystems, StatsBase, PyPlot

# Equation 13 in Pincus (1991)
function eom_henon(x, p, n)
    R = p[1]
    x, y = (x...,)
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

for (i, L) in enumerate(ts_lengths)
    k = 1
    while k <= nreps
        sys = henon(u₀ = rand(2), R = 0.8)
        t = trajectory(sys, L, Ttr = 5000)

        if !any([containsinf(tᵢ) for tᵢ in t])
            x, y = columns(t)
            apen = approx_entropy(x, r = 0.05, m = 2)
            apens_08[i][k] = apen
            k += 1
        end
    end
end

f = figure(figsize = (6, 5))

# First subplot is an example time series
sys = henon(u₀ = [0.5, 0.1], R = 0.8)
x, y = columns(trajectory(sys, 100, Ttr = 500))

subplot(211)
plot(x, label = "x")
plot(y, label = "y")
xlabel("Time (t)")
ylabel("Value")

# Second subplot contains the approximate entropy values
subplot(212)
boxplot(apens_08, positions = ts_lengths, widths = [150, 150, 150, 150], notch = true)
scatter(ts_lengths, [0.337, 0.385, NaN, 0.394], label = "Pincus (1991)")
xlabel("Time series length (L)")
ylabel("ApEn(m = 2, r = 0.05)")

legend()
tight_layout()
savefig("approx_entropy_pincus.png") # hide
```

![Approximate entropy](approx_entropy_pincus.png)
