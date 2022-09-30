
@test_throws ArgumentError approx_entropy(Dataset(rand(100, 3)))

# "Analytical tests" - compare with results from the `approximateEntropy` function
# in MATLAB (which uses natural logarithms)
x = [0, 1, 0, 1, 0, 1, 0] # A regular signal that should have approx entropy close to zero
m, τ, r  = 2, 2, 0.5
res_x_matlab = -0.036497498714443
res_x_ent = approx_entropy(x; r, m, τ, base = MathConstants.e)
@test round(res_x_ent, digits = 5) == round(res_x_matlab, digits = 5)

y = repeat([collect(-0.5:0.1:0.5); collect(0.4:-0.1:-0.4)], 5)
m, τ, r = 2, 1, 0.3
res_y_matlab = 0.195753687224351
res_y_ent = approx_entropy(y; r, m, τ, base = MathConstants.e)
@test round(res_y_ent, digits = 5) == round(res_y_matlab, digits = 5)
