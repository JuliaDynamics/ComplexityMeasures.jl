using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
pest = CountOccurrences()

h = information(JackknifeEstimator(Shannon()), pest, x)
@test h isa Real
@test h >= 0.0

# Since the plugin estimator is a lower bound for the Shannon entropy (see Arora et al.;
# https://arxiv.org/pdf/2204.01469.pdf), we should get a larger or equal value with the
# jackknife estimator.
@test h >= information(PlugIn(Shannon()), pest, x)
