using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
pest = UniqueElements()

h = information(MillerMadow(Shannon()), pest, x)
@test h isa Real
@test h >= 0.0

# Since the plugin estimator is a lower bound for the Shannon entropy (see Arora et al.;
# https://arxiv.org/pdf/2204.01469.pdf), we should get a larger or equal value with the
# Miller-Madow estimator.
@test h >= information(PlugIn(Shannon()), pest, x)

# The correction (m - 1)/2N is in nats, and is converted to the base of the entropy.
cts = counts(pest, x)
m = count(>(0), cts)
for base in (2, ℯ, 10)
    hp = information(PlugIn(Shannon(; base)), pest, x)
    hm = information(MillerMadow(Shannon(; base)), pest, x)
    @test hm - hp ≈ (m - 1) / (2 * length(x)) / log(base)
end

# N is the number of encoded points: here the number of ordinal patterns, which is
# smaller than `length(y)`.
y = randn(rng, 30)
o = OrdinalPatterns(m = 3)
cts = counts(o, y)
@test sum(cts) == 28
hp = information(PlugIn(Shannon(; base = ℯ)), o, y)
hm = information(MillerMadow(Shannon(; base = ℯ)), o, y)
@test hm - hp ≈ (count(>(0), cts) - 1) / (2 * 28)
