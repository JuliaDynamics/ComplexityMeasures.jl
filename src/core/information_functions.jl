export information, information_maximum, information_normalized, convert_logunit
export entropy

###########################################################################################
# Discrete
###########################################################################################
# Using an `InformationMeasure` directly is the same as using a `PlugIn` information
# estimator.
const InfoMeasureOrEst = Union{InformationMeasure, DiscreteInfoEstimator}

# Using an `OutcomeSpace` directly is the same as using a `RelativeAmount` probabilities estimator.
const ProbEstOrOutcomeSpace = Union{OutcomeSpace, ProbabilitiesEstimator}

"""
    information(e::DiscreteInfoEstimator, [est::ProbabilitiesEstimator,] o::OutcomeSpace, x) → h::Real

Estimate a discrete information measure from input data `x` using the provided
[`DiscreteInfoEstimator`](@ref) and [`ProbabilitiesEstimator`](@ref) over the
given [`OutcomeSpace`](@ref).

As an alternative, you can provide an [`InformationMeasure`](@ref)
for the first argument which will default to [`PlugIn`](@ref) estimation) for
the information estimation. You may also skip the first argument, in which case
`Shannon()` will be used. You may also skip
the second argument (`est`) argument, giving only an outcome space, which will default to the
[`RelativeAmount`](@ref) probabilities estimator.
Note that some information measure estimators (e.g., [`GeneralizedSchuermann`](@ref))
operate directly on counts and hence ignore `est`.


    information([e::DiscreteInfoEstimator,] p::Probabilities) → h::Real
    information([e::DiscreteInfoEstimator,] c::Counts) → h::Real

Like above, but estimate the information measure from the pre-computed
[`Probabilities`](@ref) `p` or [`Counts`](@ref).
Counts are converted into probabilities using [`RelativeAmount`](@ref),
unless the estimator `e` uses counts directly.

See also: [`information_maximum`](@ref), [`information_normalized`](@ref)
for a normalized version.

## Examples (naive estimation)

The simplest way to estimate a discrete measure is to provide the
[`InformationMeasure`](@ref) directly in combination with an [`OutcomeSpace`](@ref).
This will use the "naive" [`PlugIn`](@ref) estimator for the measure, and the "naive"
[`RelativeAmount`](@ref) estimator for the probabilities.

```julia
x = randn(100) # some input data
o = ValueBinning(RectangularBinning(5)) # a 5-bin histogram outcome space
h_s = information(Shannon(), o, x)
```

Here are some more examples:

```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = information(ps) # gives 1, about 1 bit by definition (Shannon entropy by default)
h = information(Shannon(), ps) # syntactically equivalent to the above
h = information(Shannon(), UniqueElements(), x) # syntactically equivalent to above
h = information(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
h = information(OrdinalPatterns(;m=3), x) # gives about 2, again by definition
```

## Examples (bias-corrected estimation)

It is known that both [`PlugIn`](@ref) estimation for information measures and
[`RelativeAmount`](@ref) estimation for probabilities are biased. The
scientific literature abounds with estimators that correct for this bias, both on the
measure-estimation level and on the probability-estimation level.
We thus provide the option to use any [`DiscreteInfoEstimator`](@ref) in combination with
any [`ProbabilitiesEstimator`](@ref) for improved estimates. Note that custom
probabilites estimators will only work with counting-compatible [`OutcomeSpace`](@ref).

```julia
x = randn(100)
o = ValueBinning(RectangularBinning(5))

# Estimate Shannon entropy estimation using various dedicated estimators
h_s = information(MillerMadow(Shannon()), RelativeAmount(), o, x)
h_s = information(HorvitzThompson(Shannon()), Shrinkage(), o, x)
h_s = information(Schuermann(Shannon()), Shrinkage(), o, x)

# Estimate information measures using the generic `Jackknife` estimator
h_r = information(Jackknife(Renyi()), Shrinkage(), o, x)
j_t = information(Jackknife(TsallisExtropy()), BayesianRegularization(), o, x)
j_r = information(Jackknife(RenyiExtropy()), RelativeAmount(),  x)
```
"""
function information(e::InformationMeasure, o::OutcomeSpace, x)
    return information(PlugIn(e), RelativeAmount(), o, x)
end
function information(e::InformationMeasure, est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    return information(PlugIn(e), est, o, x)
end
function information(e::DiscreteInfoEstimator, o::OutcomeSpace, x)
    return information(e, RelativeAmount(), o, x)
end
function information(e::Union{InformationMeasure, DiscreteInfoEstimator}, c::Counts)
    return information(e, probabilities(c))
end

# dispatch for `information(e, ps::Probabilities)`
# is in the individual information definition or discrete estimator files

# Convenience
information(o::OutcomeSpace, x) = information(Shannon(), o, x)
information(probs::Probabilities) = information(Shannon(), probs)
information(c::Counts) = information(Shannon(), c)

# from before https://github.com/JuliaDynamics/ComplexityMeasures.jl/pull/239
"""
    entropy(args...)

`entropy` is nothing more than a call to [`information`](@ref) that will
simply throw an error if used with an information measure that is not an entropy.
"""
function entropy(args...)
    e = first(args)
    # Check the condition for throwing an error (if false)
    cond = if e isa ProbEstOrOutcomeSpace
        # Shannon is used as default information measure
        true
    elseif e isa InformationMeasure
        # Any subtype of entropy
        e isa Entropy
    elseif e isa InformationMeasureEstimator
        # Estimator is for any subtype of entropy
        e.definition isa Entropy
    elseif e isa Probabilities
        # Default to Shannon entropy if no information measure is given
        return information(Shannon(), args...)
    else
        false
    end
    cond || throw(ArgumentError("""
        You have used `entropy` without an entropy definition
        ($(typeof(e))). Use `information` instead."""))
    return information(args...)
end


###########################################################################################
# Normalize API
###########################################################################################
"""
    information_maximum(e::InformationMeasure, o::OutcomeSpace [, x])

Return the maximum value of the given information measure can have, given input data `x`
and the given outcome space (the [`OutcomeSpace`](@ref) may also be specified by a
[`ProbabilitiesEstimator`](@ref)).

Like in [`outcome_space`](@ref), for some outcome spaces, the possible outcomes are known
without knowledge of input `x`, in which case the function dispatches to
`information_maximum(e, o)`.

    information_maximum(e::InformationMeasure, L::Int)

The same as above, but computed directly from the number of total outcomes `L`.
"""
function information_maximum(e::InformationMeasure, est::OutcomeSpace, args...)
    L = total_outcomes(est, args...)
    return information_maximum(e, L)
end
function information_maximum(e::InformationMeasure, ::Int)
    throw(ErrorException("not implemented for entropy type $(nameof(typeof(e)))."))
end
function information_maximum(e::InformationMeasureEstimator, args...)
    information_maximum(e.definition, args...)
end

"""
    information_normalized([e::DiscreteInfoEstimator,] [est::ProbabilitiesEstimator,] o::OutcomeSpace, x) → h::Real

Estimate the normalized version of the given discrete information measure,
This is just the value of [`information`](@ref) divided its maximum possible value given `o`.

The same convenience syntaxes as in [`information`](@ref) can be used here.

Notice that there is no method
`information_normalized(e::DiscreteInfoEstimator, probs::Probabilities)`,
because there is no way to know
the number of _possible_ outcomes (i.e., the [`total_outcomes`](@ref)) from `probs`.

## Normalized values

For the [`PlugIn`](@ref) estimator, it is guaranteed that `h̃ ∈ [0, 1]`. For any other
estimator, we can't guarantee this, since the estimator might over-correct. You should know
what you're doing if using anything but [`PlugIn`](@ref) to estimate normalized values.
"""
function information_normalized(e::DiscreteInfoEstimator, est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    # If the maximum information is zero (i.e. only one outcome), then we define
    # normalized information as 0.0.
    infomax = information_maximum(e, o, x)
    if infomax == 0
        return 0.0
    end
    return information(e, est, o, x) / infomax
end

function information_normalized(o::OutcomeSpace, x)
    return information_normalized(Shannon(), RelativeAmount(), o, x)
end
function information_normalized(e::InformationMeasure, o::OutcomeSpace, x)
    return information_normalized(PlugIn(e), RelativeAmount(), o, x)
end
function information_normalized(e::InformationMeasure, est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    return information_normalized(PlugIn(e), est, o, x)
end
function information_normalized(e::DiscreteInfoEstimator, o::OutcomeSpace, x)
    return information_normalized(e, RelativeAmount(), o, x)
end
function information_normalized(e::DiscreteInfoEstimator, c::Counts)
    return information_normalized(e, probabilities(c))
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


###########################################################################################
# Utils
###########################################################################################
"""
    log_with_base(base) → f

Return a function that computes the logarithm at a given base.
This definitely increases accuracy, and probably also performance.
"""
function log_with_base(base)
    if base == 2
        log2
    elseif base == MathConstants.e
        log
    elseif base == 10
        log10
    else
        x -> log(base, x)
    end
end

"""
    convert_logunit(h_a::Real, base_from, base_to) → h_b

Convert a number `h_a` computed with logarithms to base `base_from` to an entropy `h_b`
computed with logarithms to base `base_to`.
This can be used to convert the "unit" of an entropy.
"""
function convert_logunit(h_a::Real, base_from, base_to)
    h_a / log(base_from, base_to)
end
