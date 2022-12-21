# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)

npts = 1000000
ea = entropy(Shannon(; base = 2), Vasicek(m = 100), rand(npts))
ea_n = entropy(Shannon(; base = ℯ), Vasicek(m = 100), randn(npts))
@test round(ea, digits = 2) == U
@test round(ea_n, digits = 2) == N

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), Vasicek(), x)
