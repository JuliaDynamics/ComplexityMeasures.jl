using Distances
using ComplexityMeasures: ProbabilitiesEstimator

export StatisticalComplexity, entropy_complexity, entropy_complexity_curves

"""
    StatisticalComplexity <: ComplexityEstimator
    StatisticalComplexity(; kwargs...)

An estimator for the statistical complexity and entropy, originally by
[Rosso2007](@cite) and generalized by [Rosso2013](@citet).

Our implementation extends the generalization to any valid distance metric, 
any [`OutcomeSpace`](@ref) with a priori known [`total_outcomes`](@ref), 
any [`ProbabilitiesEstimator`](@ref), and any normalizable discrete 
[`InformationMeasure`](@ref).

Used with [`complexity`](@ref).

## Keyword arguments

- `o::OutcomeSpace = OrdinalPatterns{3}()`. The [`OutcomeSpace`](@ref), which controls how 
    the input data are discretized.
- `pest::ProbabilitiesEstimator = RelativeAmount()`: The
    [`ProbabilitiesEstimator`](@ref) used to estimate probabilities over the discretized
    input data.
- `hest = Renyi()`: An [`InformationMeasure`](@ref) of choice. Any information
    measure that defines [`information_maximum`](@ref) is valid here. The measure will
    be estimated using the [`PlugIn`](@ref) estimator. For example, you can use
    `hest = Renyi()`. Typically, for the information measure, an entropy such as 
    [`Shannon`](@ref) or [`Renyi`](@ref) is used. However, other measures can be used too,
    for example extropies like  [`ShannonExtropy`](@ref) (which were not treated in
    [Rosso2013](@citet)). 
- `dist<:SemiMetric = JSDivergence()`: The distance measure (from Distances.jl) to use for
    estimating the distance between the estimated probability distribution and a uniform
    distribution with the same maximal number of outcomes.

## Description

Statistical complexity is defined as

```math
C_q[P] = \\mathcal{H}_q\\cdot \\mathcal{Q}_q[P],
```

where ``Q_q`` is a "disequilibrium" obtained from a distance-measure and
``H_q`` a disorder measure.
In the original paper[Rosso2007](@cite), this complexity measure was defined
via an ordinal pattern-based probability distribution (see [`OrdinalPatterns`](@ref)), 
using [`Shannon`](@ref) entropy as the information measure, and the Jensen-Shannon 
divergence as a distance measure.

Our implementation is a further generalization of the complexity measure developed in 
[Rosso2013](@citet). We let ``H_q``be any normalizable [`InformationMeasure`](@ref), e.g. 
[`Shannon`](@ref), [`Renyi`](@ref) or [`Tsallis`](@ref) entropy, and we let
 ``Q_q`` be either on the Euclidean, Wooters, Kullback, q-Kullback, Jensen or q-Jensen 
 distance as

```math
Q_q[P] = Q_q^0\\cdot D[P, P_e],
```

where ``D[P, P_e]`` is the distance between the obtained distribution ``P``
and a uniform distribution with the same maximum number of bins, measured by
the distance measure `dist`.

## Usage

The statistical complexity is exclusively used in combination with the chosen information
measure (typically an entropy). The estimated value of the information measure can be
accessed as a `Ref` value of the struct as

```julia
x = randn(100)
c = StatisticalComplexity()
compl = complexity(c, x)
entr = c.entr_val[]
```

`complexity(c::StatisticalComplexity, x)` returns only the statistical complexity.
To obtain both the value of the entropy (or other information measure) and the
statistical complexity together as a `Tuple`, use the wrapper [`entropy_complexity`](@ref).

See also: [`entropy_complexity_curves`](@ref).
"""
struct StatisticalComplexity{D, 
        H <: DiscreteInfoEstimator{<:InformationMeasure},
        E <: ProbabilitiesEstimator, 
        O <: OutcomeSpace} <: ComplexityEstimator
    dist::D 
    hest::H
    pest::E
    o::O
    entr_val::Base.RefValue{T} where T
end

# ----------------------------------------------------------------
# Pretty printing (see /core/pretty_printing.jl).
# ----------------------------------------------------------------
oneline_printing(::Type{<:StatisticalComplexity}) = false
hidefields(::Type{<:StatisticalComplexity}) = [:entr_val]


function StatisticalComplexity(; 
        dist::D = JSDivergence(), 
        hest::H = PlugIn(Renyi()), 
        pest::E = RelativeAmount(), 
        o::O = OrdinalPatterns{3}(), 
        entr_val = Ref(0.0), kwargs...) where {D, H, E, O}
    if hest isa InformationMeasure
        hest = PlugIn(hest)
    end
    @assert hest isa DiscreteInfoEstimator
    # Deprecations. Since type dispatch doesn't operate on keywords, we need to put 
    # the deprecations inside the default constructor. For deprecations to work, 
    # 
    if haskey(kwargs, :entr)
        msg = "Keyword argument `entr` is deprecated. Use `hest` instead. " * 
         "Since you used `entr`, any value you gave `hest` will be overridden."
        @warn msg
        hest = PlugIn(kwargs[:entr])
    end

    if haskey(kwargs, :est)
        msg = "Keyword argument `est` is deprecated. " * 
            "Use `o` to specify the outcome space instead. " *
            "Since you used `est`, any value you gave `pest` will be overridden. " * 
            "Note: the probabilities estimator `pest` must be provided separately ";
        @warn msg
        o = kwargs[:est]
    end

    return StatisticalComplexity(dist, hest, pest, o, entr_val)
end



function complexity(c::StatisticalComplexity, x)
    p = allprobabilities(c.pest, c.o, x)
    return complexity(c, p)
end

"""
    entropy_complexity(c::StatisticalComplexity, x) → (h, compl)

Return a information measure `h` and the corresponding
[`StatisticalComplexity`](@ref) value `compl`.

Useful when wanting to plot data on the "entropy-complexity plane".
See also [`entropy_complexity_curves`](@ref).
"""
function entropy_complexity(c::StatisticalComplexity, x)
    compl = complexity(c, x)
    return (c.entr_val[], compl)
end

function complexity(c::StatisticalComplexity, p::Probabilities)
    L = total_outcomes(c.o)
    if length(p) != L
        throw(ArgumentError(
            "`p` must contain the probabilities for every outcome in Ω, but contains only $(length(p))
            out of $L outcomes.
            If you are trying to call `complexity(::StatisticalComplexity, p::Probabilities)`,
            you must set `p = allprobabilities(probest, outcomespace, x)`."
            ))
    end
    H_q = information(c.hest, p) / information_maximum(c.hest, c.o)

    # calculate distance between calculated distribution and uniform one
    D_q = evaluate(c.dist, vec(p), fill(1.0/L, L))

    # generate distribution with just one filled bin
    deterministic = zeros(L)
    deterministic[1] = 1

    D_max = evaluate(c.dist, deterministic, fill(1.0/L, L))
    C_q = D_q / D_max * H_q
    c.entr_val[] = H_q

    return C_q
end

linearpermissiverange(start; stop, length) = length==1 ? [start] : collect(range(start, stop=stop, length=length))

"""
    entropy_complexity_curves(c::StatisticalComplexity; 
        num_max=1, num_min=1000) -> (min_entropy_complexity, max_entropy_complexity)

Calculate the maximum complexity-entropy curve for the statistical complexity according to
[Rosso2007](@citet) for `num_max * total_outcomes(c.o)` different values of the normalized
information measure of choice (in case of the maximum complexity curves)
and `num_min` different values of the normalized information measure of choice (in case of
the minimum complexity curve).

This function can also be used to compute the maximum "complexity-extropy curve" if
`c.hest` is e.g. [`ShannonExtropy`](@ref), which is the equivalent of the
complexity-entropy curves, but using extropy instead of entropy. 

## Description

The way the statistical complexity is designed, there is a minimum and maximum possible
complexity for data with a given value of an information measure.
The calculation time of the maximum complexity curve grows as `O(total_outcomes(c.o)^2)`,
and thus takes very long for high numbers of outcomes.
This function is inspired by S. Sippels implementation in statcomp [Sippel2016](@cite).

This function will work with any `ProbabilitiesEstimator` where [`total_outcomes`](@ref)
is known a priori.
"""
function entropy_complexity_curves(c::StatisticalComplexity; num_max::Int = 1, num_min::Int=1000)

    L = total_outcomes(c.o)
    # avoid having to resize later by just making result containers vectors straight away.
    hs_cs_max = zeros(SVector{2, Float64}, (L-1)*num_max)

    p = Probabilities(zeros(L); normed = true) # can't normalize zeros, so let's pretend this is already normalized
    prob_params = linearpermissiverange(0; stop = 1 / L, length = num_max)

    j = 1
    for i in 1:(L - 1)
        vec(p) .= 0.0 # Note 0.0, not 0 (the elements in `p` are floats, so we should re-fill with floats to avoid conversions)
        for k in 1:num_max
            # Does this function ensure sum(p) == 1? If not, we need to normalize `p` afterwards, because `information` requires
            # normalized probabilities (i.e. summing to 1)
            _fill_probs_k!(p, prob_params, L, i, k)
            compl = complexity(c, p)
            hs_cs_max[j] = SVector(c.entr_val[], compl)
            j += 1
        end
    end
    hs = [x[1] for x in hs_cs_max]
    args = sortperm(hs)
    hs_cs_max = hs_cs_max[args]

    prob_params = linearpermissiverange(1/L; stop=1, length=num_min)
    hs_cs_min = zeros(SVector{2, Float64}, num_min)
    p = ones(L)

    for i in 1:num_min
        fill!(p, 1.0)
        p .*= (1-prob_params[i]) / (L-1)
        p[1] = prob_params[i]
        probs = Probabilities(p, true)
        compl = complexity(c, probs)
        hs_cs_min[end-i+1] = SVector(c.entr_val[], compl)
    end
    return (
        hs_cs_min,
        hs_cs_max
    )
end

function _fill_probs_k!(p, prob_params, L, i, k)
    probs = vec(p)
    probs[1] = prob_params[k] # why set first element here if overwriting it in the loop below?
    # if we know that p has sufficient length relative to L and i,
    # @inbounds can save some computation time by skipping bounds checking.
    @inbounds for j = 1:(L - i)
        probs[j] = (1 - prob_params[k]) / (L - i)
    end
end
