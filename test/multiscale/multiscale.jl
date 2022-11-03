est = Dispersion()
x = rand(100)
maxscale = 5
e = Shannon()

# Generic tests is all we need here. The tests that make sure that the entropies are
# computed for the correctly sampled timeseries are in `/test/multiscale/downsampling.jl`
mr = multiscale(e, Regular(),  x, est; maxscale)
mrn = multiscale_normalized(e, Regular(), x, est; maxscale)
@test mr isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mrn) == 5

mc = multiscale(e, Composite(), x, est; maxscale)
mcn = multiscale_normalized(e, Composite(), x, est; maxscale)
@test mc isa Vector{T} where T <: Real
@test mcn isa Vector{T} where T <: Real
@test length(mc) == 5
@test length(mcn) == 5
