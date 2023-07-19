# Constructors
m1 = RenyiExtropy(q = 2)
m2 = RenyiExtropy(2)
@test m1.q == 2
@test m2.q == 2

# Minimized for one-element distributions, where there is total order.
x = [0.1, 0.1, 0.1]
@test information(RenyiExtropy(q = 2), CountOccurrences(), x) == 0.0
@test information_normalized(RenyiExtropy(q = 2), CountOccurrences(), x) == 0.0


# Example 3.2 from Liu & Xiao (2021) (equivalence of Rényi entropy/extropy for
# two-element distributions)
x = [0.2, 0.4, 0.4]
h = information(Renyi(; q = 2, base = MathConstants.e), CountOccurrences(), x)
j = information(RenyiExtropy(; q = 2, base = MathConstants.e), CountOccurrences(), x)
@test round(h, digits = 4) ≈ round(j, digits = 4) ≈ 0.5878

# Example 3.1 from Liu & Xiao (2021) (equivalence of Rényi entropy/extropy for
# two-element distributions). They use natural logs - otherwise the numbers don't
# make any sense.
x = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6]
h = information(Renyi(; q = 2, base = MathConstants.e), CountOccurrences(), x)
j = information(RenyiExtropy(; q = 2, base = MathConstants.e), CountOccurrences(), x)
@test round(h, digits = 4) ≈ 1.0986
@test round(j, digits = 4) ≈ 0.8109

# In general, normalized Rényi extropy should be maximized (i.e. be equal to 1) for a
# uniform distribution.
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
jr = information_normalized(RenyiExtropy(; q = 2), CountOccurrences(), x)
@test jr ≈ 1

# The maximum Rényi extropy should be insensitive to the base of the logarithm.
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
j2 = RenyiExtropy(; q = 2, base = 2)
j10 = RenyiExtropy(; q = 2, base = 10)
est = CountOccurrences()
@test information_normalized(j2, est, x) ≈ information_normalized(j10, est, x)

# The Rényi extropy should equal the Shannon extropy for q = 1
jq1 = RenyiExtropy(; q = 2, base = 2)
@test information(jq1, est, x) ≈ information(ShannonExtropy(; base = 2), est, x)
