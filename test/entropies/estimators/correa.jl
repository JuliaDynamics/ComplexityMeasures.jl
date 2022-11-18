# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)

ec = Correa(m = 100, base = 2)
ec_n = Correa(m = 100, base = MathConstants.e)

n = 1000000
@test round(entropy(ec, rand(rng, n)), digits = 2) == U
@test round(entropy(ec_n, randn(rng, n)), digits = 2) == N

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), Correa(), x)
