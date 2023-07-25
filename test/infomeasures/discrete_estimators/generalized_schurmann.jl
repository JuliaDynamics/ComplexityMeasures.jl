using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
pest = CountOccurrences()

h_singlea = information(GeneralizedSchürmann(Shannon(); a = 1.0), pest, x)
@test h_singlea isa Real
@test h_singlea >= 0.0

# `x` has five outcomes, so we need five parameters
as = fill(0.5, 5)
h_multiplea = information(GeneralizedSchürmann(Shannon(); a = as), pest, x)
@test h_multiplea isa Real

# In general, the correction yields different results.
@test h_singlea != h_multiplea

# Mean triplet Shannon entropy estimates; appears just after eq. 28 in Grassberger (2002).
nreps = 1000
hs = zeros(nreps)
hest = GeneralizedSchürmann(Shannon(; base = 2)) # results are given in bits
for i = 1:nreps
    hs[i] = information(hest, pest, rand(0:1, 3))
end
@show mean(hs)


############################################################################################
# Mean triplet Shannon entropy estimates; appears just after eq. 28 in Grassberger (2002).
############################################################################################
nreps = 2_000_000
hs = zeros(nreps)
#hest = GeneralizedSchürmann(Shannon(; base = 2)) # results are given in bits
pest = CountOccurrences()
for i = 1:nreps
    hs[i] = information(PlugIn(Shannon()), pest, rand(rng, 0:1, 3))
end
@test round(mean(hs), digits = 3) == 0.689

# Compare
hs = zeros(nreps)
hest = GeneralizedSchürmann(Shannon(; base = 2); a = 1) # results are given in bits
pest = CountOccurrences()

for i = 1:nreps
    hs[i] = information(hest, pest, rand(rng, 0:1, 3))
end
mean(hs)
@test round(mean(hs), digits = 3) == 1.000
