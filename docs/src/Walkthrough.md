# Walkthrough entropy

```@docs
WalkthroughEntropy
walkthrough_entropy
```

## Examples

Here, we reproduce parts of Fig. 1 from Stoop et al. (2021).

We start by creating some symbolic time series of length `N`. Then, because we're 
generating the walkthrough entropy for multiple positions ``n``, we use
`entropygenerator` for each time series, so that initialization computations only happens 
once per time series. Finally, we compute the walkthrough entropy at all positions `1:N`.

```@example
using Entropies, PyPlot

N = 1200
# Generate time series
x_lowfreq = "a"^(N ÷ 3) * "b"^(N ÷ 3) * "c"^(N ÷ 3);
x_hifreq = "abc"^(N ÷ 3)
x_rw3 = rand(['a', 'b', 'c'], N)
x_rw2 = rand(['a', 'b'], N)
x_a = "a"^(N)
x_ab_2 = "ab"^(N ÷ 2)
x_ab_20 = ("a"^20*"b"^20)^(N ÷ 40)
x_ab_200 = ("a"^200*"b"^200)^(N ÷ 400)

# Initialize entropy generators
method = WalkthroughEntropy()
e_lofreq = entropygenerator(x_lowfreq, method);
e_hifreq = entropygenerator(x_hifreq, method);
e_rw3 = entropygenerator(x_rw3, method);
e_rw2 = entropygenerator(x_rw2, method);
e_a = entropygenerator(x_a, method);
e_ab_2 = entropygenerator(x_ab_2, method);
e_ab_20 = entropygenerator(x_ab_20, method);
e_ab_200 = entropygenerator(x_ab_200, method);

# Compute walkthrough entropies through positions 1:N
base = MathConstants.e
hs_lofreq = [e_lofreq(i, base = base) for i = 1:N]
hs_hifreq = [e_hifreq(i, base = base) for i = 1:N]
hs_wn3 = [e_rw3(i, base = base) for i = 1:N]
hs_wn2 = [e_rw2(i, base = base) for i = 1:N]
hs_a = [e_a(i, base = base) for i = 1:N]
hs_ab_2 = [e_ab_2(i, base = base) for i = 1:N]
hs_ab_20 = [e_ab_20(i, base = base) for i = 1:N]
hs_ab_200 = [e_ab_200(i, base = base) for i = 1:N]

# Plot
ns = (1:N |> collect) ./ N
unit = "nats"


f = figure(figsize = (10,7))
ax = subplot(231)
plot(xlabel = "n/N", ylabel = "h [$unit]");
plot(ns, hs_hifreq, label = "abcabc...")
plot(ns, hs_lofreq, label = "aa...bb..ccc")
xlabel("n/N")
ylabel("h ($unit)")
legend()

ax = subplot(232)
plot(xlabel = "n/N", ylabel = "h [$unit]");
plot(ns, hs_wn3, label = "RW (k = 3)")
xlabel("n/N")
ylabel("h ($unit)")
legend()

ax = subplot(234)
plot(ns, hs_a, label = "k = 1")
plot(ns, hs_ab_2, label = "k = 2, T = 2")
plot(ns, hs_ab_20, label = "k = 2, T = 20")
xlabel("n/N")
ylabel("h ($unit)")
legend()

ax = subplot(235)
plot(ns, hs_a, label = "k = 1")
plot(ns, hs_ab_2, label = "k = 2, T = 2")
plot(ns, hs_ab_200, label = "k = 2, T = 200")
xlabel("n/N")
ylabel("h ($unit)")
legend()

ax = subplot(236)
plot(ns, hs_wn2, label = "RW (k = 2)")
plot(ns, hs_ab_2, label = "k = 2, T = 2")
xlabel("n/N")
ylabel("h ($unit)")

legend()
tight_layout()
PyPlot.savefig("walkthrough_entropy.png")
```

![Walkthrough entropy](walkthrough_entropy.png)
