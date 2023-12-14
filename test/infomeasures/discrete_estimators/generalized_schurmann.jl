using Test, Random
using ComplexityMeasures
using Statistics: mean
rng = MersenneTwister(1234)

x = rand(rng, 1:5, 1000)
ospace = UniqueElements()

h_singlea = information(GeneralizedSchürmann(a = 1.0), ospace, x)
@test h_singlea ≥ 0.0

# Since the plugin estimator is a lower bound for the Shannon entropy (see Arora et al.;
# https://arxiv.org/pdf/2204.01469.pdf), we should get a larger or equal value with the
# generalized Schürmann estimator.
@test h_singlea ≥ information(PlugIn(Shannon()), ospace, x)


# `x` has five outcomes, so we need five parameters
as = fill(0.5, 5)
h_multiplea = information(GeneralizedSchürmann(a = as), ospace, x)
@test h_multiplea isa Real

# In general, the correction yields different results.
@test h_singlea != h_multiplea

# Since the plugin estimator is a lower bound for the Shannon entropy (see Arora et al.;
# https://arxiv.org/pdf/2204.01469.pdf), we should get a larger or equal value with the
# generalized Schürmann estimator.
@test h_multiplea ≥ information(PlugIn(Shannon()), ospace, x)

# Mean triplet Shannon entropy estimates; appears just after eq. 28 in Grassberger (2002).
nreps = 1000
hs = zeros(nreps)
hest = GeneralizedSchürmann(Shannon(; base = 2)) # results are given in bits
for i = 1:nreps
    hs[i] = information(hest, ospace, rand(0:1, 3))
end

############################################################################################
# Mean triplet Shannon entropy estimates; appears just after eq. 28 in Grassberger (2002).
############################################################################################
nreps = 2_000_000
hs = zeros(nreps)
#hest = GeneralizedSchürmann(Shannon(; base = 2)) # results are given in bits
ospace = UniqueElements()
for i = 1:nreps
    hs[i] = information(PlugIn(Shannon()), ospace, rand(rng, 0:1, 3))
end
@test round(mean(hs), digits = 2) == 0.69

# Compare
hs = zeros(nreps)
hest = GeneralizedSchürmann(Shannon(; base = 2); a = 1) # results are given in bits
ospace = UniqueElements()

for i = 1:nreps
    hs[i] = information(hest, ospace, rand(rng, 0:1, 3))
end
mean(hs)
@test round(mean(hs), digits = 3) == 1.000
