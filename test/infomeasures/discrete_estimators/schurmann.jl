using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
pest = CountOccurrences()

h_singlea = information(Schürmann(Shannon(); ξ = 1), pest, x)
@test h_singlea isa Real
@test h_singlea >= 0.0
