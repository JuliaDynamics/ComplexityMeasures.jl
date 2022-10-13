est = Dispersion()
x = rand(100)
e = Shannon()

# TODO: test actual outcomes too. Make some analytical examples.
@test multiscale(e, Regular(),  x, est) isa Vector{T} where T <: Real
@test multiscale(e, Composite(),  x, est) isa Vector{T} where T <: Real
