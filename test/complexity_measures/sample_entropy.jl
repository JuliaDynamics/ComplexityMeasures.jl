# Analytical examples seem to be lacking in the literature. As a next-best-test,
# we just check that we get a sample sufficiently close to zero for a completely
# regular signal.
N = 6000
x = repeat([-5:5 |> collect; 4:-1:-4 |> collect], N ÷ 20);
se = sample_entropy(x; m = 2, τ = 1, r = 0.1)
@test round(se, digits = 3) == round(0.0, digits = 3)

# Conversely, a non-regular signal should result in a sample entropy
# greater than zero.
x = rand(N)
se = sample_entropy(x; m = 2, τ = 1, r = 0.1)
@test round(se, digits = 2) > round(0.0, digits = 2)
