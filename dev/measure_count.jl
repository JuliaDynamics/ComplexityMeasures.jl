# # [Counting the number of measures in ComplexityMeasures.jl](@id total_measures)

# In this page we will count all the possible complexity measures
# than one can compute with the current version of ComplexityMeasures.jl!

using ComplexityMeasures
using InteractiveUtils: subtypes
import Pkg; Pkg.status("ComplexityMeasures")

# First let's define a function that counts concrete subtypes
# that we will be re-using to count measures in ComplexityMeasures.jl.

concrete_subtypes(type::Type) = concrete_subtypes!(Any[], type)
function concrete_subtypes!(out, type::Type)
    if !isabstracttype(type)
        push!(out, type)
    else
        foreach(T -> concrete_subtypes!(out, T), subtypes(type))
    end
    out
end

# ## Count Based Outcome Spaces
#
# Each `OutcomeSpace` is a possible way of discretizing the input data. For the
# purpose of counting measures, **we treat the an outcome space with different
# input parameters as the same outcome space overall.**

# Some outcome spaces are count-based. We estimate these separately,
# because they may be estimated with various different probabilities estimators.
# For our counting here it doesn't matter whether the outcome space supports
# spatiotemporal or trajectory (uni/multi variate) date.
# We only care if it is counting based or not.

OUTCOME_SPACES_COUNT = concrete_subtypes(ComplexityMeasures.CountBasedOutcomeSpace)

# We do a small correction here because two outcome spaces
# aren't count-based but for internal convenience they satisfy
# the subtyping relationship
correction_ospaces = (AmplitudeAwareOrdinalPatterns, WeightedOrdinalPatterns)
foreach(
    T -> deleteat!(OUTCOME_SPACES_COUNT, findfirst(isequal(T), OUTCOME_SPACES_COUNT)),
    correction_ospaces
)

OUTCOME_SPACES_COUNT

# Probabilities can be estimated from count-based outcome spaces in different
# ways. We count the same `ProbabilitiesEstimators` with different input
# parameters as the same estimator. Each probabilities estimator can be
# combined with any outcome space.

PROBESTS_COUNT = concrete_subtypes(ProbabilitiesEstimator)

# and we count the combinations

n_outcome_spaces_count = length(OUTCOME_SPACES_COUNT)
n_probests_count = length(PROBESTS_COUNT)
n_probs_count = n_outcome_spaces_count * n_probests_count

# ## Non-count-based outcome spaces
#
# We also provide some outcome spaces that are not count-based, but can still
# be used to estimate discrete probabilities by using some sort of "relative
# amount" estimation.

OUTCOME_SPACES_NOCOUNT = setdiff(
    concrete_subtypes(ComplexityMeasures.OutcomeSpace),
    concrete_subtypes(ComplexityMeasures.CountBasedOutcomeSpace),
)

# to which we add back the outcome spaces correction
push!(OUTCOME_SPACES_NOCOUNT, correction_ospaces...)
OUTCOME_SPACES_NOCOUNT

# Only `RelativeAmount` probabilities estimator works with non-count-based outcome spaces
n_probs_noncount = length(OUTCOME_SPACES_NOCOUNT) * 1

# ## Grand total of extracting PMFs from data

# Therefore the total ways to estimate discrete probabilities from data
# in ComplexityMeasures.jl is just

n_probs_discrete = n_probs_noncount + n_probs_count

# ## Discrete Information measures

# Currently, the InformationMeasures implemented are different types of
# entropies and the lesser-known extropies. Each of these measures, in their
# discrete form, are functions of probability mass functions (PMFs).
# Therefore, we can combine each of these measure with probabilities estimated
# using any count-based outcome space and any probabilities estimator.

# Let's collect all of these discrete measures

INFO_MEASURES_DISCRETE = concrete_subtypes(InformationMeasure)

# Each information measure can be estimated using any of the generic
# estimators:

INFO_MEASURE_ESTIMATOR_GENERIC = concrete_subtypes(ComplexityMeasures.DiscreteInfoEstimatorGeneric)

# so we count by multiplying

n_discrete_infoest_generic = length(INFO_MEASURES_DISCRETE)*length(INFO_MEASURE_ESTIMATOR_GENERIC)


# In addition to the generic estimators,
# we also provide additional estimators specific to Shannon entropy.
INFO_MEASURE_ESTIMATOR_SHANNON = concrete_subtypes(ComplexityMeasures.DiscreteInfoEstimatorShannon)

# For these there is no variability of the information measure so
n_discrete_estimators_shannon = length(INFO_MEASURE_ESTIMATOR_SHANNON)

# This gives us the total possible ways of estimating information measures
# given a PMF:

n_discrete_info_est = n_discrete_estimators_shannon + n_discrete_infoest_generic

# ## Grand total of discrete information measures

# This total is obtained as the direct multiplication of all ways
# to obtain a PMF and all ways to compute an information measure from PMF

n_discrete_info = n_discrete_info_est * n_probs_discrete

# That's quite a lot and we are only half-way done!

# ## Differential information measures

# The differential information measures and their estimators are
# all grouped into one level of abstraction as long as the user is concerned,
# so counting things here is very simple!

DIFF_INFO_EST = concrete_subtypes(DifferentialInfoEstimator)

# All of these estimate one quantity (the differential Shannon entropy),
# with the exception of one particular estimator (`LeonenkoProzantoSavani`)
# that can estimate also Tsallis and Renyi entropies.
# Therefore, the number of differential measures one can estimate within
# ComplexityMeasures.jl is:

n_diff_info = length(DIFF_INFO_EST) + 2

# ## Complexity measures
#
# We also provide a number of estimators that are not probability based, which
# we call just complexity estimators for this discussion.
# We count each of these as a separate measure.

COMPLEXITY_ESTIMATORS = concrete_subtypes(ComplexityEstimator)

# However, from these we need to treat `StatisticalComplexity` separately, so we have

n_complexity_measures_basic = length(COMPLEXITY_ESTIMATORS) - 1

# In ComplexityMeasures.jl `StatisticalComplexity` can be combined with any discrete
# information measure, any information estimator, and any count-based outcome space.
# Additionally, `StatisticalComplexity` in ComplexityMeasures.jl can be combined with any
# metric from the Distances.jl package.

# For `StatisticalComplexity`, counting all possible combinations of outcome spaces,
# probabilities estimators, information measure definitions, information measure estimators,
# along with distance measures, as unique measures would over-inflate the measure count.
# For practicality, we here count different version of `StatisticalComplexity` by considering
# the number of statistical complexity measures resulting from counting unique outcome spaces
# and information measures, since these are the largest contributors to changes in the
# computed numerical value of the measure. Therefore we have:

n_complexity_measures_statistical_complexity = length(INFO_MEASURES_DISCRETE) * length(concrete_subtypes(OutcomeSpace))

# which gives us the following total number of complexity estimators

n_complexity_measures_total = n_complexity_measures_basic + n_complexity_measures_statistical_complexity

# ## Probabilities functions

# Besides calculating complexity measures, ComplexityMeasures.jl gives the user
# the unique possibility of accessing the probability mass function directly.
# As we show in the associated article, this allows rather straightforwardly
# defining new, or expanding existing, complexity measures. For example,
# the `MissingDispersionPatterns` is just a wrapper of the `missing_outcomes` function.

# Therefore, we believe it is fair to count a couple of probabilities functions by
# themselves as additional complexity measures. In particular, we count here
# the functions `probabilities, allproabilities` as candidates for it,
# as all other functions of the library are simple processing of these two.
# Given that each function can work with any type of outcome space
# and probability estimation technique, we obtain

n_extra_prob_measures = 2 * n_probs_discrete

# ## Grand total of measures

# Right, so the grand total of all measures that can be estimated with
# ComplexityMeasures.jl are:

n_grand_total =
  n_discrete_info +
  n_diff_info +
  n_complexity_measures_total +
  n_extra_prob_measures