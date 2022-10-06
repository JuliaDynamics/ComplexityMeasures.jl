# Create an analytical test from scratch.
x = [
    1 1 1;
    2 2 2;
    3 3 1;
]
stencil = [1 1; 1 0]


est = SpatialDispersion(stencil, x, symbolization = GaussianSymbolization(c = 3),
    periodic = false)

# With n = 3 categories, a GaussianSymbolization should not alter this particular `x`.
# we thus expect the following "dispersion patterns" [frequencies]:
# "121" [2]
# "232" [2],
# so we should get a uniform two-element probabilitity distribution.
@test probabilities_and_events(x, est) == ([0.5, 0.5], ["121", "232"])

# With periodic boundary conditions, we expect a different dispersion pattern distribution.
# `probabilities` sorts the dispersion patterns, so we must also consider frequencies of
# patterns sorted lexicographically.
est = SpatialDispersion(stencil, x, symbolization = GaussianSymbolization(c = 3),
    periodic = true)
# "113" [1]
# "121" [3]
# "212" [1]
# "232" [2]
# "311" [1]
# "313" [1]
@test probabilities_and_events(x, est) == (
    [1/9, 3/9, 1/9, 2/9, 1/9, 1/9],
    ["113", "121", "212", "232", "311", "313"]
    )

# Normalized Shannon entropy should be close to 1 when the obtained probability
# distribution is close to uniform. This happens for uniform noise.
# --------------------------------------------------------------------------------
# The following data `y` has 2 classes. c = 2 ensures that this remains the case after
# symbolization. We can also skip symbolization all together, but then we must specify
# `L` as the total possible number of symbols the input data can take.
y = rand(0:1, 100, 100);
est_y = SpatialDispersion(stencil, y, symbolization = GaussianSymbolization(c = 2))
est_y_presymb = SpatialDispersion(stencil, y; skip_symbolization = true, L = 2)

@test 0.99 <= round(entropy_normalized(y, est_y), digits = 2) <= 1.0
@test 0.99 <= round(entropy_normalized(y, est_y_presymb), digits = 2) <= 1.0
