using DelayEmbeddings: genembed
using DelayEmbeddings: Dataset

m = 4
τ = 1
τs = tuple([τ*i for i = 0:m-1]...)
x = rand(250)
D = genembed(x, τs)
est = Kraskov(k = 3, w = 1)
e = Shannon()
er = Renyi(q = 1.5)
@test_throws ArgumentError entropy(er, est, D)


@test entropy(est, D) isa Real
@test entropy(e, est, D) isa Real

# Analytical test.
XN = Dataset(randn(100000, 1));
# For normal distribution with mean 0 and std 1, the entropy is
h_XN_base_e = 0.5 * log(MathConstants.e, 2π) + 0.5 # nats
h_XN_base_2 = h_XN_base_e / log(2, MathConstants.e) # bits

h_XN_kr_base_e = entropy(Kraskov(k = 3, base = MathConstants.e), XN)
h_XN_kr_base_2 = entropy(Kraskov(k = 3, base = 2), XN)
@test round(h_XN_base_e, digits = 1) == round(h_XN_kr_base_e, digits = 1)
@test round(h_XN_base_2, digits = 1) == round(h_XN_kr_base_2, digits = 1)

@test_throws ArgumentError entropy(Renyi(q = 2), Kraskov(), XN)
