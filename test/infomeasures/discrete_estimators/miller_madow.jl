using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
pest = CountOccurrences()

h = information(MillerMadow(Shannon()), pest, x)
@test h isa Real
@test h >= 0.0
