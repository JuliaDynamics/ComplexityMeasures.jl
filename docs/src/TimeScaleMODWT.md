# Time-scale (wavelet)

```@docs
TimeScaleMODWT
```

## Example

The scale-resolved wavelet entropy should be lower for very regular signals (most of the 
energy is contained at one scale) and higher for very irregular signals (energy spread
more out across scales).

```@example
using Entropies, PyPlot
N, a = 1000, 10
t = LinRange(0, 2*a*π, N)

x = sin.(t);
y = sin.(t .+  cos.(t/0.5));
z = sin.(rand(1:15, N) ./ rand(1:10, N))

est = TimeScaleMODWT()
h_x, h_y, h_z = genentropy(x, est), genentropy(y, est), genentropy(z, est)

f = figure(figsize = (10,6))
ax = subplot(311)
px = plot(t, x; color = "C1", label = "h=$(h=round(h_x, sigdigits = 5))"); 
ylabel("x"); legend()
ay = subplot(312)
py = plot(t, y; color = "C2", label = "h=$(h=round(h_y, sigdigits = 5))"); 
ylabel("y"); legend()
az = subplot(313)
pz = plot(t, z; color = "C3", label = "h=$(h=round(h_z, sigdigits = 5))"); 
ylabel("z"); xlabel("Time"); legend()
tight_layout()
savefig("waveletentropy.png")
```

![](waveletentropy.png)