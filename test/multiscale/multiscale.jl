est = Dispersion()
x = rand(100)
maxscale = 5

e = Shannon()
mr = multiscale(e, Regular(),  x, est; maxscale, normalize = false)
mc = multiscale(e, Composite(),  x, est; maxscale, normalize = false)
mrn = multiscale(e, Regular(),  x, est; maxscale, normalize = true)
mcn = multiscale(e, Composite(),  x, est; maxscale, normalize = true)
@test mr isa Vector{T} where T <: Real
@test mc isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test mcn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mc) == 5
@test length(mrn) == 5
@test length(mcn) == 5

c = ReverseDispersion()
mcr = multiscale(c, Regular(), x; maxscale, normalize = false)
mcc = multiscale(c, Composite(), x; maxscale, normalize = false)
mcrn = multiscale(c, Regular(), x; maxscale, normalize = true)
mccn = multiscale(c, Composite(), x; maxscale, normalize = true)
@test mcr isa Vector{T} where T <: Real
@test mcc isa Vector{T} where T <: Real
@test mcrn isa Vector{T} where T <: Real
@test mccn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mc) == 5
@test length(mrn) == 5
@test length(mcn) == 5

# TODO: To verify that downsampling algorithms work as expected, we need to make a few
# simple analytical examples too. There are no such examples in the original papers,
# so we'll have to make some ourselves.
