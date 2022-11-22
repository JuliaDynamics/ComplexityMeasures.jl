using DelayEmbeddings: Dataset

# To ensure minimal rectangle volumes are correct, we also test internals directly here.
# It's not feasible to construct an end-product test due to the neighbor searches.
x = Dataset([[-1, -2], [0, -2], [3, 2]])
y = Dataset([[3, 1], [-5, 1], [3, -2]])
@test Entropies.volume_minimal_rect([0, 0], x) == 24
@test Entropies.volume_minimal_rect([0, 0], y) == 40

# Analytical tests for the estimated entropy
DN = Dataset(randn(200000, 1))
hN_base_e = 0.5 * log(MathConstants.e, 2π) + 0.5
hN_base_2 = hN_base_e / log(2, MathConstants.e)

est = Zhu(k = 3)

@test round(entropy(Shannon(; base = ℯ), est, DN), digits = 1) == round(hN_base_e, digits = 1)
@test round(entropy(Shannon(; base = 2), est, DN), digits = 1) == round(hN_base_2, digits = 1)

@test_throws ArgumentError entropy(Renyi(q = 2), Zhu(), DN)

# Shannon entropy is default.
@test entropy(Shannon(; base = 2), est, DN) ==  entropy(est, DN, base = 2)
@test entropy(Shannon(; base = ℯ), est, DN) ==  entropy(est, DN, base = ℯ)
