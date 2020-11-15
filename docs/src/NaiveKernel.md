# Kernel density

```@docs
NaiveKernel
```

## Distance evaluation methods

```@docs
TreeDistance
DirectDistance
```

## Example

Here, we draw some random points from a 2D normal distribution. Then, we use kernel 
density estimation to associate a probability to each point `p`, measured by how many 
points are within radius `1.5` of `p`. Plotting the actual points, along with their 
associated probabilities estimated by the KDE procedure, we get the following surface 
plot.

```@example
using Distributions, PyPlot, DelayEmbeddings, Entropies
𝒩 = MvNormal([1, -4], 2)
N = 500
D = Dataset(sort([rand(𝒩) for i = 1:N]))
x, y = columns(D)
p = probabilities(D, NaiveKernel(1.5))
surf(x, y, p.p)
xlabel("x"); ylabel("y")
savefig("kernel_surface.png")
```

![](kernel_surface.png)
