###########################################################################################
# Discrete
###########################################################################################
# Using an `InformationMeasure` directly is the same as using a `PlugIn` information
# estimator.
const InfoMeasureOrEst = Union{InformationMeasure, DiscreteInfoEstimator}

# Using an `OutcomeSpace` directly is the same as using a `RelativeAmount` probabilities estimator.
const ProbEstOrOutcomeSpace = Union{OutcomeSpace, ProbabilitiesEstimator}

"""
    information([e::DiscreteInfoEstimator,] est::ProbabilitiesEstimator, x) → h::Real

Estimate a discrete information measure from input data `x` using the provided
[`DiscreteInfoEstimator`](@ref) and [`ProbabilitiesEstimator`](@ref).
As an alternative, you can provide an [`InformationMeasure`](@ref)
for the first argument (which will default to [`PlugIn`](@ref) estimation) or an
[`OutcomeSpace`](@ref) for the second argument (which will default to the [`RelativeAmount`](@ref)
estimator).

All estimators compute [`Shannon`](@ref) entropy by default. To estimate other measures,
give them as an argument to the estimator, e.g.
`information(Jackknife(Renyi()), probest, x)`.
If `e` is not provided, then the default is `information(PlugIn(Shannon()), probs)`.

## Examples (naive estimation)

The simplest way to estimate a discrete measure is to provide the
[`InformationMeasure`](@ref) directly in combination with an [`OutcomeSpace`](@ref).
This will use the "naive" [`PlugIn`](@ref) estimator for the measure, and the "naive"
[`RelativeAmount`](@ref) estimator for the probabilities.

```julia
x = randn(100) # some input data
o = ValueHistogram(RectangularBinning(5)) # a 5-bin histogram outcome space
h_s = information(Shannon(), o, x)
```

Here are some more examples:

```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = information(ps) # gives 1, about 1 bit by definition (Shannon entropy by default)
h = information(Shannon(), ps) # syntactically equivalent to the above
h = information(Shannon(), CountOccurrences(), x) # syntactically equivalent to above
h = information(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
h = information(SymbolicPermutation(;m=3), x) # gives about 2, again by definition
```

## Examples (bias-corrected estimation)

It is known that both [`PlugIn`](@ref) and [`RelativeAmount`](@ref) estimation are biased. The
scientific literature abounds with estimators that correct for this bias, both on the
measure-estimation level and on the probability-estimation level.
We thus provide the option to use any [`DiscreteInfoEstimator`](@ref) in combination with
any [`ProbabilitiesEstimator`](@ref) for improved estimates. Note that custom
probabilites estimators will only work with counting-compatible [`OutcomeSpace`](@ref).

```julia
x = randn(100)
o = ValueHistogram(RectangularBinning(5))

# Estimate Shannon entropy estimation using various dedicated estimators
h_s = information(MillerMadow(Shannon()), RelativeAmount(o), x)
h_s = information(HorvitzThompson(Shannon()), Shrinkage(o), x)
h_s = information(Schürmann(Shannon()), Shrinkage(o), x)

# Estimate information measures using the generic `Jackknife` estimator
h_r = information(Jackknife(Renyi()), Shrinkage(o), x)
j_t = information(Jackknife(TsallisExtropy()), BayesianRegularization(o), x)
j_r = information(Jackknife(RenyiExtropy()), RelativeAmount(o), x)
```

## Maximum/normalized information

Most discrete [`InformationMeasure`](@ref)s have a well-defined maximum for a given
[`OutcomeSpace`](@ref). You can use [`information_maximum`](@ref) to get this
maximum value, and [`information_normalized`](@ref) to obtain a normalized version of the
measure (divided by the maximum value).


    information([e::DiscreteInfoEstimator,] p::Probabilities) → h::Real

Like above, but estimate the information measure from the pre-computed
[`Probabilities`](@ref) `p`.

See also: [`information_maximum`](@ref), [`information_normalized`](@ref).
"""
function information(e::InformationMeasure, o::OutcomeSpace, x)
    return information(PlugIn(e), RelativeAmount(o), x)
end
function information(e::InformationMeasure, est::ProbabilitiesEstimator, x)
    return information(PlugIn(e), est, x)
end
function information(e::DiscreteInfoEstimator, o::OutcomeSpace, x)
    return information(e, RelativeAmount(o), x)
end

# dispatch for `information(e, ps::Probabilities)`
# is in the individual information definition or discrete estimator files

# Convenience
information(est::OutcomeSpace, x) = information(Shannon(), est, x)
information(probs::Probabilities) = information(Shannon(), probs)


###########################################################################################
# Normalize API
###########################################################################################
"""
    information_maximum(e::InformationMeasure, o::OutcomeSpace, x)
    information_maximum(e::InformationMeasure, est::ProbabilitiesEstimator, x)

Return the maximum value of the given information measure can have, given input data `x`
and  the given outcome space (the [`OutcomeSpace`](@ref) may also be specified by a
[`ProbabilitiesEstimator`](@ref)).

Like in [`outcome_space`](@ref), for some outcome spaces, the possible outcomes are known
without knowledge of input `x`, in which case the function dispatches to
`information_maximum(e, est)`.

    information_maximum(e::InformationMeasure, L::Int)

The same as above, but computed directly from the number of total outcomes `L`.
"""
function information_maximum(e::InformationMeasure, est::ProbEstOrOutcomeSpace, x)
    L = total_outcomes(est, x)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasure, est::ProbEstOrOutcomeSpace)
    L = total_outcomes(est)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasure, ::Int)
    throw(ErrorException("not implemented for entropy type $(nameof(typeof(e)))."))
end
function information_maximum(e::InformationMeasureEstimator, args...)
    information_maximum(e.definition, args...)
end

"""
    information_normalized([e::DiscreteInfoEstimator,] o::OutcomeSpace, x) → h̃
    information_normalized([e::DiscreteInfoEstimator,] est::ProbabilitiesEstimator, x) → h̃

Estimate `h̃`, a normalized discrete information measure, from input data `x`, using the
[`DiscreteInfoEstimator`](@ref) `e`. This is just the value of [`information`](@ref)
divided by the maximum value for `e`, according to
the given [`OutcomeSpace`](@ref) (which may be specified by `est` if not given directly).

Instead of a discrete information measure estimator, an [`InformationMeasure`](@ref)
can be given as first argument, in which case [`PlugIn`](@ref) estimation is used.
 If `e` is not given, it defaults to `Shannon()`.

Notice that there is no method
`information_normalized(e::DiscreteInfoEstimator, probs::Probabilities)`,
because there is no way to know
the number of _possible_ outcomes (i.e., the [`total_outcomes`](@ref)) from `probs`.

## Normalized values

For the [`PlugIn`](@ref) estimator, it is guaranteed that `h̃ ∈ [0, 1]`. For any other
estimator, we can't guarantee this, since the estimator might over-correct. You should know
what you're doing if using anything but [`PlugIn`](@ref) to estimate normalized values.

"""
function information_normalized(e::InformationMeasure, o::OutcomeSpace, x)
    # If the maximum information is zero (i.e. only one outcome), then we define
    # normalized information as 0.0.
    infomax = information_maximum(e, o, x)
    if infomax == 0
        return 0.0
    end
    return information(e, o, x) / infomax
end
function information_normalized(o::OutcomeSpace, x::Array_or_SSSet)
    return information_normalized(Shannon(), o, x)
end



##########################################################################################
# Differential
##########################################################################################
"""
    information(est::DifferentialInfoEstimator, x) → h::Real

Estimate a **differential information measure** using the provided
[`DifferentialInfoEstimator`](@ref) and input data `x`.

## Description

The overwhelming majority of differential estimators estimate the [`Shannon`](@ref)
entropy. If the same estimator can estimate different information measures (e.g. it can
estimate both [`Shannon`](@ref) and [`Tsallis`](@ref)), then the information measure
is provided as an argument to the estimator itself.

See the [table of differential information measure estimators](@ref table_diff_ent_est)
in the docs for all differential information measure estimators.

Currently, unlike for the discrete information measures, this method doesn't involve
explicitly first computing a probability density function and then passing this density
to an information measure definition. But in the future, we want to establish a
`density` API similar to the [`probabilities`](@ref) API.

## Examples

To compute the differential version of a measure, give it as the first argument to a
[`DifferentialInfoEstimator`](@ref) and pass it to [`information`](@ref).

```julia
x = randn(1000)
h_sh = information(Kraskov(Shannon()), x)
h_vc = information(Vasicek(Shannon()), x)
```

A normal distribution has a base-e Shannon differential entropy of
`0.5*log(2π) + 0.5` nats.

```julia
est = Kraskov(k = 5, base = ℯ) # Base `ℯ` for nats.
h = information(est, randn(2_000_000))
abs(h - 0.5*log(2π) - 0.5) # ≈ 0.0001
```
"""
function information(est::DifferentialInfoEstimator, args...)
    throw(ArgumentError("`information` not implemented for $(nameof(typeof(est)))"))
end

# Dispatch for these functions is implemented in individual estimator files in
# `differential_info_estimators/`.

function information(::InformationMeasure, ::DifferentialInfoEstimator, args...)
    throw(ArgumentError(
        """`InformationMeasure` must be given as an argument to `est`, not to `information`.
        """
    ))
end
