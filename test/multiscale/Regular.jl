using Entropies

est = Dispersion()
x = rand(100)
maxscale = 5
e = Shannon()

# Generic tests is all we need here. The tests that make sure that the entropies are
# computed for the correctly sampled timeseries are in `/test/multiscale/downsampling.jl`
mr = Entropies.multiscale(Regular(), e, est, x; maxscale)
mrn = Entropies.multiscale_normalized(Regular(), e, est, x; maxscale)
@test mr isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mrn) == 5

# `DiffEntropyEst`s` should work for `multiscale`, but not `multiscale_normalized`
@test Entropies.multiscale(Regular(), e, Kraskov(), x) isa Vector{T} where T <: Real
@test_throws ErrorException Entropies.multiscale_normalized(Regular(), e, Kraskov(), x)
