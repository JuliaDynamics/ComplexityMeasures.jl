# # Tutorial

# The goal of this tutorial is threefold:

# 1. To convey the _terminology_ used by ComplexityMeasures.jl: key terms, what they mean, and how they are used within the codebase
# 2. To provide a _rough overview_ of the overall features provided by ComplexityMeasures.jl
# 3. To introduce the _main API functions_ of ComplexityMeasures.jl in a single, self-contained document: how these functions connect to key terms, what are their main inputs and outputs, and how they are used in realistic scientific scripting

# !!! note
#     The documentation and exposition of ComplexityMeasures.jl is inspired by chapter 5 of
#     [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
#     Datseris & Parlitz, Springer 2022, and expanded to cover more content.


# ## First things first: "complexity measures"

# "Complexity measure" is a generic, umbrella term, used extensively in the nonlinear timeseries analysis (NLTS) literature.
# Roughly speaking, a complexity measure is a quantity extracted from input data that quantifies some dynamical property in the data (often, complexity measures are entropy variants).
# These complexity measures can highlight some aspects of the dynamics more than others, or distinguish one type of dynamics from another, or classify timeseries into classes with different dynamics, among other things.
# ComplexityMeasures.jl implements hundreds such measures and hence it is named as such. To enable this, ComplexityMeasures.jl is more than a collection "dynamic statistics": it is also a framework for rigorously defining probability spaces and estimating probabilities from input data.

# Within the codebase of ComplexityMeasures.jl we make a separation with the functions [`information`](@ref) (or its daughter function [`entropy`](@ref)) and [`complexity`](@ref).
# We use `information` for complexity measures that are _explicit functionals of probability mass or probability density functions_, even though these measures might not be labelled as "information measures" in the literature.
# We use `complexity` for other complexity measures that are not explicit functionals of probabilities.
# We stress that the separation between `information` and `complexity` is purely pragmatic, to establish a generic and extendable software interface within ComplexityMeasures.jl.

# ## The basis: Probabilities and Outcome Spaces

# Information measures and some other complexity measures are computed based on **probabilities** derived from input data.
# In order to derive probabilities from data, an **outcome space** (also called a sample space) needs to be defined: a way to transform data into elements $\omega$ of an outcome space $\omega \in \Omega$, and assign probabilities to each outcome $p(\omega)$, such that $p(\Omega)=1$. $\omega$ are called _outcomes_ or _events_.
# In code, outcome spaces are subtypes of [`OutcomeSpace`](@ref).
# For example, one outcome space is the [`ValueHistogram`](@ref), which is the most commonly known outcome space, and corresponds to discretizing data by putting the data values into bins of a specific size.

using ComplexityMeasures

x = randn(10_000)
ε = 0.1 # bin width
o = ValueHistogram(ε)
o isa OutcomeSpace

# such outcome spaces may be given to [`probabilities`](@ref) to estimate the corresponding probabilities.

probs = probabilities(o, x)

# In this example the probabilities are the (normalized) heights of each bin of the histogram. However, we don't know the bins, which are the _elements_ of the outcome space. To obtain the probabilities and the bins we would use

probs, outs = probabilities_and_outcomes(o, x)
outs

# here the outcomes are the left edges of each bin. This allows us to straightforwardly visualize the results

using CairoMakie
left_edges = first.(outs) # covert `Vector{SVector}` into `Vector{Real}`
barplot(left_edges, probs; axis = (ylabel = "probability",))

# Naturally, there are other outcome spaces one may use, and one can find the list of implemented ones in [`OutcomeSpace`](@ref).
# A prominent example used in the NLTS literature are ordinal patterns.
# The outcome space for it is [`SymbolicPermutation`](@ref), and can be particularly useful with timeseries that come from nonlinear dynamical systems. For example, if we simulate a logistic map timeseries

using DynamicalSystemsBase

logistic_rule(u, r, t) = SVector(r*u[1]*(1 - u[1]))
ds = DeterministicIteratedMap(logistic_rule, [0.4], 4.0)
Y, t = trajectory(ds, 10_000; Ttr = 100)
y = Y[:, 1]
summary(y)

# we can estimate the probabilities corresponding to the ordinal patterns

o = SymbolicPermutation()
probsy, outsy = probabilities_and_outcomes(o, y)
hcat(probsy, outsy)

# and compare them with those for the purely random timeseries `x`:

probsx, outsx = probabilities_and_outcomes(o, x)
hcat(probsx, outsx)

# You will notice that there are more outcomes for the `x` timeseries than the `y`. All _possible_ outcomes, i.e., the cardinality of the outcome space, can be found with [`total_outcomes`](@ref)

total_outcomes(o)

# The reason for less outcomes in the `y` results is that one was never encountered in the `y` data.
# This is a common theme in ComplexityMeasures.jl: outcomes that are not in the data are skipped.
# This can save huge amounts of memory for outcome spaces with very large numbers of outcomes.
# To explicitly obtain all outcomes, by assigning 0 probability to not encountered outcomes, use [`allprobabilities`](@ref) or [`allprobabilities_and_outcomes`](@ref).
# For [`SymbolicPermutation`](@ref) the outcome space does not depend on input data and is always the same.
# Hence, the corresponding outcomes matching to [`allprobabilities`](@ref), coincide for `x` and `y`, and also coincide with the output of the function [`outcome_space`](@ref):

o = SymbolicPermutation()

probsx = allprobabilities(o, x)
probsy = allprobabilities(o, y)
outsx = outsy = outcome_space(o)

hcat(probsx, probsy, outsx)

# So far we have been estimating probabilities by counting the amount of times each possible outcome was encountered in the data.
# This simplified approach is called "maximum likelihood estimation".
# The direct counts themselves may be obtained using [`counts`](@ref)

countsy = counts(o, y)
probsy = probabilities(o, y)
hcat(countsy, countsy ./ sum(countsy), probsy)

# By definition columns 2 and 3 are identical.
# However, there are other ways to estimate probabilities that may account for biases in counting outcomes from finite data. Alternative estimators for probabilities are subtypes of [`ProbabilitiesEstimator`](@ref).
# `ProbabilitiesEstimator`s wrap outcome space instances and dictate alternative ways to estiamte probabilities.
# For example, one could use [`BayesianRegularization`](@ref)

probsy_bayes = probabilities(BayesianRegularization(o), y)

probsy_bayes .- probsy

# While the corrections of [`BayesianRegularization`](@ref) are small, they are nevertheless measurable. In truth, when calling [`probabilities`](@ref) with an outcome space instance, the default [`RelativeAmount`](@ref) probabilities estimator is used to extract the probabilities.

# ## Entropies

# Many compexity measures are a straightforward estimation of Shannon entropy with a given outcome space.
# For example, the well known _permutation entropy_ is exactly the Shannon entropy of the probabilities `probsy` we computed above based on ordinal patterns.

perm_ent_x = entropy(SymbolicPermutation(), x)
perm_ent_y = entropy(SymbolicPermutation(), y)
(perm_ent_x, perm_ent_y)

# As expected, the permutation entropy of the `x` signal is higher, because the signal is "more random".

# Just like above with the possibility of many different outcome spaces, the same concept applies to entropy. There are many different entropies, Shannon is not the only one!
# Each entropy is a subtype of [`EntropyDefinition`](@ref).
# Another commonly used entropy is the Renyi or generalized entropy.
# We can use [`Renyi`](@ref) as an additional first argument to the [`entropy`](@ref) function

perm_ent_y_q2 = entropy(Renyi(;q = 2.0), SymbolicPermutation(), y)
(perm_ent_y_q2, perm_ent_y)

# In fact, when we called `entropy(SymbolicPermutation(), y)`, this dispatch to the default call of `entropy(Shannon(), SymbolicPermutation(), y)`.

# ## More than entropies: estimators and other information measures

# The estimation of an entropy truly parallelizes the estimation of probabilities: in the latter, we could decide an outcome space _and_ an _estimator_ to estimate probabilities.
# The same happes for entropy: we can decide an entropy definition and an _estimator_ of how to estimate the entropy. For example, instead of the default [`PlugIn`](@ref) estimator that we used above implicitly, we could use [`Jackknife`](@ref)

ospace = SymbolicPermutation()
entdef = Renyi(;q = 2.0)
entest = Jackknife(entdef)
perm_ent_y_q2_jack = entropy(entest, ospace, y)

(perm_ent_y_q2, perm_ent_y_q2_jack)

# It is up to the researcher to read the documentation of the plethora of estimators implemented and decide what is most suitable for their data at hand. They all can be found in [`DiscreteInfoEstimator`](@ref).

# Now, recall that at the very beginning of this notebook we mentioned a code separation of [`information`](@ref) and [`complexity`](@ref).
# We did this because there are other measures, besides entropy, that are explicit functionals of some probability mass function.
# One example is the Shannon _extropy_ [`ShannonExtropy`](@ref), the completent of _entropy_, which could be computed as

extdef = ShannonExtropy()
perm_ext_y = information(extdef, ospace, y)

# and, just like the Shannon _entropy_, it could also be estimated with a different estimator such as [`Jackknife`](@ref)

perm_ext_y_jack = information(Jackknife(extdef), ospace, y)

# In truth, when we called `entropy(e, o, y)` it dispatched automatically to `information(e, o, y)`, as all "information measures" are part of the same function interface.

# ## Beyond discrete: differential entropies


# ## Beyond information: other complexity measures

# As discussed at the very beginning of this tutorial, there are some complexity measures that are not explicit functionals of probabilities, and hence cannot be straightforwardly related to an outcome space, in the sense of providing an instance of [`OutcomeSpace`](@ref) to the estimation function.