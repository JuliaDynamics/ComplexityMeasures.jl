# Sample entropy

```@docs
sample_entropy
```

```@example
using Entropies, PyPlot, Distributions
N = 2000
x_𝒰 = rand(N)
x_𝒩 = rand(Normal(0, 3), N)
x_periodic = repeat(rand(20), N ÷ 20)

x_𝒰 .= (x_𝒰 .- mean(x_𝒰)) ./ std(x_𝒰)
x_𝒩 .= (x_𝒩 .- mean(x_𝒩)) ./ std(x_𝒩)
x_periodic .= (x_periodic .- mean(x_periodic)) ./ std(x_periodic)

rs = 10 .^ range(-1, 0, length = 30)
base = 2
m = 2
hs_𝒰 = [sample_entropy(x_𝒰, m = m, r = r, base = base) for r in rs]
hs_𝒩 = [sample_entropy(x_𝒩, m = m, r = r, base = base) for r in rs]
hs_periodic = [sample_entropy(x_periodic, m = m, r = r, base = base) for r in rs]

f = figure(figsize = (4, 4))
subplot(111)
plot(rs, hs_𝒰, label = "Uniform noise, U(0, 1)")
plot(rs, hs_𝒩, label = "Gaussian noise, N(0, 1)")
plot(rs, hs_periodic, label = "Periodic signal")
xlabel("r")
ylabel("h")
legend()
xscale("log")
tight_layout()
PyPlot.savefig("sample_entropy.png") # hide
```

![Sample entropy](sample_entropy.png)
