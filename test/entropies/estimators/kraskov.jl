# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = round((0.5*log(2π) + 0.5) / log(3, ℯ), digits = 2) # custom base

npts = 1000000
ea = entropy(Shannon(; base = 2), Kraskov(k = 5), rand(npts))
ea_n = entropy(Shannon(; base = ℯ), Kraskov(k = 5), randn(npts))
ea_n3 = entropy(Shannon(; base = 3), Kraskov(k = 5), randn(npts))

@test round(ea, digits = 2) == U
@test round(ea_n, digits = 2) == N
@test round(ea_n3, digits = 2) == N_base3

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), Kraskov(), x)
