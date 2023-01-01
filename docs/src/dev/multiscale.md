
## Complexity: multiscale

```@example
using ComplexityMeasures
using CairoMakie

N, a = 2000, 20
t = LinRange(0, 2*a*π, N)

x = repeat([-5:5 |> collect; 4:-1:-4 |> collect], N ÷ 20);
y = sin.(t .+ cos.(t/0.5)) .+ 0.2 .* x
maxscale = 10
hs = ComplexityMeasures.multiscale_normalized(Regular(), SampleEntropy(y), y; maxscale)

fig = Figure()
ax1 = Axis(fig[1,1]; ylabel = "y")
lines!(ax1, t, y; color = Cycled(1));
ax2 = Axis(fig[2, 1]; ylabel = "Sample entropy (h)", xlabel = "Scale")
scatterlines!(ax2, 1:maxscale |> collect, hs; color = Cycled(1));
fig
```
