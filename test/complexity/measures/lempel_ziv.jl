using Random
# Example from
# Amigó, J. M., Szczepański, J., Wajnryb, E., & Sanchez-Vives, M. V. (2004). Estimating
# the entropy rate of spike trains via Lempel-Ziv complexity. Neural Computation, 16(4),
# 717-736.
# This should give the subsequences: 0, 1, 011, 0100, 011011, 1001, 0, or 7 elements
x = [0,1,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,0]

@test complexity(LempelZiv76(), x) == 7

# A very regular signal should have normalized LempelZiv76-complexity close to zero.
ts = sin.(0:0.5:100000)
μ = mean(ts)
x = [x <= μ ? 0 : 1 for x in ts]
c = complexity_normalized(LempelZiv76(), x)
@test 0 ≤ c ≤ 0.05

# A random sequence should have normalized LempelZiv76-complexity close to 1
rng = MersenneTwister(1234)
ts = rand(10000)
μ = mean(ts)
x = [x <= μ ? 0 : 1 for x in ts]
c = complexity_normalized(LempelZiv76(), x)
@test 0.95 ≤ c ≤ 1.05
