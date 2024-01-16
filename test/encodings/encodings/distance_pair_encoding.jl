using ComplexityMeasures, Test
using DelayEmbeddings
using Distances

# ----------------------------------------------------------------
# Analytical example
# ----------------------------------------------------------------
m = Chebyshev()
y = [SVector(1.0), SVector(0.5), SVector(0.25), SVector(0.64)]
dmax = m(y[1], y[2]) # dist = 0.50
dmin = m(y[2], y[3]) # dist = 0.25
dmid = m(y[3], y[4]) # dist = 0.39

# This should give five bins with left adges at [0.25], [0.30], [0.35], [0.40] and [0.45]
encoding = PairDistanceEncoding(dmin, dmax; n = 5, metric = m)

# Ensure distances are in the correct bin.
@test encode(encoding, (y[1], y[2])) == 5
@test encode(encoding, (y[2], y[3])) == 1
@test encode(encoding, (y[3], y[4])) == 3
@test decode(encoding, encode(encoding, (y[1], y[2]))) ≈ [0.45]
@test decode(encoding, encode(encoding, (y[2], y[3]))) ≈ [0.25]
@test decode(encoding, encode(encoding, (y[3], y[4]))) ≈ [0.35]