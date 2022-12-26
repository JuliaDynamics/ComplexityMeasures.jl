# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# EntropyDefinition to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# EntropyDefinition with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = round((0.5*log(2π) + 0.5) / log(3, ℯ), digits = 2) # custom base

npts = 1000000
ea = entropy(Shannon(; base = 2), AlizadehArghami(m = 100), rand(npts))
ea_n = entropy(Shannon(; base = ℯ), AlizadehArghami(m = 100), randn(npts))
ea_n3 = entropy(Shannon(; base = 3), AlizadehArghami(m = 100), randn(npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N * 0.98 ≤ ea_n ≤ N * 1.02
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), AlizadehArghami(), x)

# Default is Shannon base-2 differential entropy
est = AlizadehArghami()
@test entropy(est, x) == entropy(Shannon(; base = 2), est, x)
