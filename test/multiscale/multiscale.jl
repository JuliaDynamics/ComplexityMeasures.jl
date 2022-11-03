est = Dispersion()
x = rand(100)
maxscale = 5
e = Shannon()

# Generic tests.
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

# TODO: To verify that downsampling algorithms work as expected, we need to make a few
# simple analytical examples too. There are no such examples in the original papers,
# so we'll have to make some ourselves.
