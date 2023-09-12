using ComplexityMeasures
using Test
using Random

x = [1 2 1; 8 3 4; 6 7 5]
# Re-create the Ribeiro et al, 2012 using stencil
# (you get 4 symbols in a 3x3 matrix. For this matrix, the upper left
# and bottom right are the same symbol. So three probabilities in the end).
stencil = CartesianIndex.([(0,0), (1,0), (0,1), (1,1)])
est = SpatialOrdinalPatterns(stencil, x; periodic = false)

# Generic tests
ps = probabilities(est, x)
@test ps isa Probabilities
@test length(ps) == 3
@test sort(ps) == [0.25, 0.25, 0.5]
h = information(Renyi(base = 2), est, x)
@test h == 1.5
# In fact, doesn't matter how we order the stencil,
# the symbols will always be equal in top-left and bottom-right
stencil = CartesianIndex.([(0,0), (1,0), (1,1), (0,1)])
est = SpatialOrdinalPatterns(stencil, x; periodic = false)
@test information(Renyi(base = 2), est, x) == 1.5

# But for sanity, let's ensure we get a different number
# for a different stencil
stencil = CartesianIndex.([(0,0), (1,0), (2,0)])
est = SpatialOrdinalPatterns(stencil, x; periodic = false)
ps = sort(probabilities(est, x))
@test ps[1] == 1/3
@test ps[2] == 2/3
# let's also check we get the same value as above
# if we specify extent and lag instead of stencil
extent = (2, 2)
lag = (1, 1)
est = SpatialOrdinalPatterns((extent, lag), x; periodic = false)
@test information(Renyi(base = 2), est, x) == 1.5

# and let's also test the matrix-way of specifying the stencil
stencil = [1 1; 1 1];
est = SpatialOrdinalPatterns(stencil, x; periodic = false)
@test information(Renyi(base = 2), est, x) == 1.5
# when the stencil is square, it is also easy to get an analytical set of outcomes.
# 1 2 1    count column major order and get vectors, starting in top left corner,
# 8 3 4    [1, 8, 2, 3], [8, 6, 3, 7], [2, 3, 1, 4], [3, 7, 4, 5], which give permutations
# 6 7 5    [1, 3, 4, 2], [3, 2, 4, 1], [3, 1, 2, 4], [1, 3, 4, 2]
ps, outs = probabilities_and_outcomes(est, x)
@test length(ps) == length(outs)
@test sort(outs) ==
    sort(SVector{4, Int}.([[1, 3, 4, 2], [3, 2, 4, 1], [3, 1, 2, 4]]))

# Also test that it works for arbitrarily high-dimensional arrays
stencil = CartesianIndex.([(0,0,0), (0,1,0), (0,0,1), (1,0,0)])
z = reshape(1:125, (5,5,5));
est = SpatialOrdinalPatterns(stencil, z; periodic = false)
# Analytically the total stencils are of length 4*4*4 = 64
# but all of them given the same probabilities because of the layout
ps = probabilities(est, z)
@test ps == [1]
# if we shuffle, we get random stuff
w = shuffle!(Random.MersenneTwister(42), collect(z))
ps = probabilities(est, w)
@test length(ps) > 1
# check that the 3d-hyperrectangle version also works as expected
# this stencil is a hyperrectangle
stencil = CartesianIndex.([(0,0,0), (0,1,0), (0,0,1),
                            (1,0,0), (0,1,1), (1,0,1),
                            (1,1,0), (1,1,1)])
est1 = SpatialOrdinalPatterns(stencil, w; periodic = false)
# which would correspond to this
extent = (2, 2, 2)
lag = (1, 1, 1)
est2 = SpatialOrdinalPatterns((extent, lag), w; periodic = false)
@test information(Renyi(), est1, w) == information(Renyi(), est2, w)

# and to this stencil written as a matrix
stencil = [1; 1;; 1; 1;;; 1; 1;; 1; 1]
est3 = SpatialOrdinalPatterns(stencil, w; periodic = false)
@test information(Renyi(), est1, w, ) == information(Renyi(), est3, w)

####################
# "Analytical" tests
####################
# We know that the normalized permutation entropy should → 1 for uniformly distributed
# noise. Test this assumption up to some tolerance.
x = rand(100, 100)
stencil = [1 1; 1 1];
est = SpatialOrdinalPatterns(stencil, x)
hsp = information_normalized(Renyi(), est, x)
println(est)
@test round(hsp, digits = 2) == 1.00

@test outcome_space(est) == outcome_space(OrdinalPatterns(m = 4))
