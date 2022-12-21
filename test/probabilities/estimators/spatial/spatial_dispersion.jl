using Entropies
using StaticArrays

# Create an analytical test from scratch.
x = [
    1 1 1;
    2 2 2;
    3 3 1;
];
stencil = [1 1; 1 0];

# Verify that constructor works as expected for different combinations of keyword arguments
@test SpatialDispersion(stencil, x) isa SpatialDispersion
@test SpatialDispersion(stencil, x, skip_encoding = false) isa SpatialDispersion
@test SpatialDispersion(stencil, x, skip_encoding = false, L = 2) isa SpatialDispersion
@test SpatialDispersion(stencil, x, skip_encoding = true, L = 2) isa SpatialDispersion
@test_throws ArgumentError SpatialDispersion(stencil, x, skip_encoding = true, L = nothing)

est = SpatialDispersion(stencil, x, c = 3, periodic = false)
# With n = 3 categories, a GaussianCDFEncoding should not alter this particular `x`.
# we thus expect the following "dispersion patterns" [frequencies]:
# "121" [2]
# "232" [2],
# so we should get a uniform two-element probabilitity distribution.
@test probabilities_and_outcomes(est, x) == (
    [0.5, 0.5],
    SVector{3, Int}.([[1, 2, 1], [2, 3, 2]]))

# With periodic boundary conditions, we expect a different dispersion pattern distribution.
# `probabilities` sorts the dispersion patterns, so we must also consider frequencies of
# patterns sorted lexicographically.
est = SpatialDispersion(stencil, x, c = 3, periodic = true)
# "113" [1]
# "121" [3]
# "212" [1]
# "232" [2]
# "311" [1]
# "313" [1]
@test probabilities_and_outcomes(est, x) == (
    [1/9, 3/9, 1/9, 2/9, 1/9, 1/9],
    SVector{3, Int}.([[1, 1, 3], [1, 2, 1], [2, 1, 2], [2, 3, 2], [3, 1, 1], [3, 1, 3]])
    )

# Normalized Shannon entropy should be close to 1 when the obtained probability
# distribution is close to uniform. This happens for uniform noise.
# --------------------------------------------------------------------------------
# The following data `y` has 2 classes. c = 2 ensures that this remains the case after
# symbolization. We can also skip symbolization all together, but then we must specify
# `L` as the total possible number of symbols the input data can take.
y = rand(0:1, 100, 100);
est_y = SpatialDispersion(stencil, y, c = 2)
est_y_presymb = SpatialDispersion(stencil, y; skip_encoding =  true, L = 2)

@test 0.99 <= round(entropy_normalized(est_y, y), digits = 2) <= 1.0
@test 0.99 <= round(entropy_normalized(est_y_presymb, y), digits = 2) <= 1.0

@test outcome_space(est_y) == outcome_space(Dispersion(c = est_y.c, m = est_y.m))
