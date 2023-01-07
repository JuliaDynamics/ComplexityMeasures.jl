# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Shannon entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Shannon entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = round((0.5*log(2π) + 0.5) / log(3, ℯ), digits = 2) # custom base

npts = 20000 # a bit fewer points than for other tests, so tests don't take forever.
ea = entropy(Shannon(), Lord(k = 20), rand(npts))
ea_n = entropy(Shannon(; base = ℯ), Lord(k = 20), randn(npts))
ea_n3 = entropy(Shannon(; base = 3), Lord(k = 20), randn(npts))

@test U - max(0.05, U*0.03) ≤ ea ≤ U + max(0.03, U*0.03)
@test N * 0.96 ≤ ea_n ≤ N * 1.03
@test N_base3 * 0.96 ≤ ea_n3 ≤ N_base3 * 1.03

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), Lord(), x)

# Default is Shannon base-2 differential entropy
est = Lord()
@test entropy(est, x) == entropy(Shannon(), est, x)
