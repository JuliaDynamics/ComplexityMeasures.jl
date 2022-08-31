
N = 100;
X = rand(1:5, N, N);
est = SymbolicPermutation2D(m = 2)
gen = entropygenerator(X, est)

@time timeit(gen)
# using BenchmarkTools
# @btime gen($X);
# Y = rand(1:5, N, N);
# @btime gen($Y);
using Entropies
using Entropies

# 50 images where each of the 10000 pixel take on binary values ("dark" or "light")
images = [rand(["dark", "light"], 100, 100) for i = 1:50]

# Create a generator using the first image as a template. All images are the same size,
# so we can re-use the generator. Use pixel blocks of size `2*2`.
est = SymbolicPermutation2D(m = 2)
eg = entropygenerator(first(images), est)

# The generalized order-`1` (Shannon) normalized permutation entropy to base 2 of each image
[eg(img, base = 2, q = 1, normalize = true) for img in images]

x = [1 2 1; 8 3 4; 6 7 5]
genentropy(x, SymbolicPermutation2D(m = 2), base = 2)

ps = [0.5, 0.25, 0.25]
genentropy(Probabilities(ps), base = 2) / log(2, factorial(2*2))
