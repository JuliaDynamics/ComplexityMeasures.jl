using DelayEmbeddings, SparseArrays
using KrylovKit: eigsolve, KrylovDefaults
using StaticArrays
using Random

include("utils.jl")

export TransferOperator,TransferOperatorApproximation, ApproximationIterative, ApproximationEigen,
    InvariantMeasure, invariantmeasure,transfermatrix, transferoperator

"""
    TransferOperator <: ProbabilitiesEstimator
    TransferOperator(approximation_method::ApproximationMethod,boundary_condition)

An [`ProbabilitiesEstimator`](@ref) based on the transfer (Perron-Frobenius) operator.

When used with [`probabilities`](@ref), then the transfer operator
is approximated using a selected outcome space, by counting occurrences of transitions between each outcome.
Probabilities are estimated as the invariant measure
associated with that transfer operator. Assumes that the input data are sequential
(time-ordered). `approximation_method` decides how the invariant measure should be calculated. See 
[`ApproximationIterative`, `ApproximationEigen`](@ref).
`boundary_condition` decides the transition from the last observed outcome: `:circular` adds a transition to the first one, 
`:random` adds a randomly chosen transition from the already observed, `:none` add nothing. 

When constructing the transfer operator from time series using `transferoperator` 
with a given `outcome_space`, a `TransferOperatorApproximation` is returned, 
giving access to the transition probabilites between the observed outcomes.

See also: [`transferoperator`, `TransferOperatorApproximation`](@ref).

## Outcome space requirements

This estimator only works with counting-compatible outcome spaces.

## Outcome ordering

Outcomes returned by [`probabilities_and_outcomes`](@ref) are ordered according to first
appearance for all outcome spaces, except for `ValueBinning`, where they returned in 
an increasing order along each axis (from left to right in 1D, left to right and 
bottom to top in 2D etc.). Thus, if

```julia
x = [1,2,3,1,2,3]
op = OrdinalPatterns{3}() #ordinal patterns 
probs,outs = probabilities_and_outcomes(TransferOperator(:circular),op,x) #use circular boundary for short time series 
```
then `probs[i]` is the invariant measure (probability) of the outcome `outs[i]`, which is
the `i`-th first appearance in the time series with nonzero measure.

```julia
x = [0.3,0.2,0.1,0.3,0.1]
vb = ValueBinning(RectangularBinning(3))

ps,outs = probabilities_and_outcomes(TransferOperator(),vb,x) #bins are ordered
```
## Description

The transfer operator ``P^{N}``is computed as an `N`-by-`N` matrix of transition
probabilities between outcomes, where `N` is the
number of observed outcomes. Note that an outcome can 
correspond to a single (ex. 1D binning) or multiple datapoints (ex. ordinal patterns).

If  ``\\mathbf{x}^(L)_n`` is the ``n``-th sequence of datapoints that is encoded 
to the ``i``-th outcome ``s_i``, and ``E(\\mathbf{x}^(L)_n) = s_i`` then 

```math
P_{ij} = \\dfrac
{\\#\\{ s_i | E(\\mathbf{x}^(L)_{n+1}) = s_j \\cap E(\\mathbf{x}^(L)_{n+1}) = s_i \\}}
{\\#\\{ s_i | E(\\mathbf{x}^(L)_{m}) = s_i \\}},
```

where ``\\#`` denotes the cardinal. The element ``P_{ij}`` thus indicates the 
transition probability from outcome ``s_i`` to ``s_j``. Thus, the row ``P_{ik}^N`` where
``k \\in \\{1, 2, \\ldots, N \\}`` gives the probability
of jumping from the outcome ``s_i`` to any of the other ``N`` outcomes. It
follows that ``\\sum_{k=1}^{N} P_{ik} = 1`` for all ``i``. Thus, ``P^N`` is a row/right
stochastic matrix.

### Invariant measure estimation from transfer operator

#### Iterative method (default)

The invariant distribution is initialized as a length-`N` random distribution which is then applied to
``P^{N}``. For reproducibility in this step, set the `rng` in `ApproximationIterative`.
The resulting length-`N` distribution is then applied to ``P^{N}`` again. This process
repeats until the difference between the distributions over consecutive iterations is
below some threshold.

Use `ApproximationIterative()` with `TransferOperator` to approximate the invariant measure 
by the eigenvector method. 

#### Eigenvector method

The left invariant distribution ``\\mathbf{\\rho}^N`` is a row vector, where
``\\mathbf{\\rho}^N P^{N} = \\mathbf{\\rho}^N``. Hence, ``\\mathbf{\\rho}^N`` is a row
eigenvector of the transfer matrix ``P^{N}`` associated with eigenvalue 1. The distribution
``\\mathbf{\\rho}^N`` approximates the invariant density of the system subject to
`outcome space`, and can be taken as a probability distribution over the 
symbolization/partition elements.

Use `ApproximationEigen()` with `TransferOperator` to approximate the invariant measure 
by the eigenvector method. 


## Precision when used with `ValueBinning` outcome space

The default behaviour when using [`RectangularBinning`](@ref) or
[`FixedRectangularBinning`](@ref) is to accept some loss of precision on the 
bin boundaries for speed-ups, but this may lead to issues for `TransferOperator`
where some points may be encoded as the symbol `-1` ("outside the binning").



!!! hint "Transfer operator approach vs. naive histogram approach"

    Why bother with the transfer operator instead of using regular histograms to obtain
    probabilities?

    In fact, the naive histogram approach and the
    transfer operator approach are equivalent in the limit of long enough time series
    (as ``n \\to \\intfy``), which is guaranteed by the ergodic theorem. There is a crucial
    difference, however:

    The naive histogram approach only gives the long-term probabilities that
    orbits visit a certain region of the state space. The transfer operator encodes that
    information too, but comes with the added benefit of knowing the *transition
    probabilities* between states (see [`transfermatrix`](@ref)).


See also: [`RectangularBinning`](@ref), [`FixedRectangularBinning`](@ref),
[`invariantmeasure`](@ref).
"""

abstract type ApproximationMethod end

struct TransferOperator <: ProbabilitiesEstimator 
    approximation_method::ApproximationMethod
    boundary_condition
end

#constructors with defaults
TransferOperator() = TransferOperator(ApproximationIterative(), :none)
TransferOperator(approximation_method::ApproximationMethod) = TransferOperator(approximation_method,:none)
TransferOperator(boundary_condition::Symbol) = TransferOperator(ApproximationIterative(), boundary_condition)

abstract type AbstractTransferOperatorApproximation <: ProbabilitiesEstimator end


"""
    TransferOperatorApproximation(transfermatrix, outcome_space::OutcomeSpace, outcomes, approximation_method)

* `transfermatrix`: an approximation to the transfer operator, subject to the
given `outcome_space`, computed over some set of sequentially ordered points.

* `outcome_space`: the outcome space that defines the outcomes 

* `outcomes`: the observed, unique outcomes  

* `approximation_method`: decides the `ApproximationMethod` used by `invariantmeasure`

Only bins actually observed outcomes are considered. 
The element `outcomes[i]` which
corresponds to the `i`-th column/row of the transfer operator `to`.

See also: [`TransferOperator`](@ref).

"""
struct TransferOperatorApproximation{OC<:OutcomeSpace,AM<:ApproximationMethod} <: AbstractTransferOperatorApproximation
    transfermatrix::AbstractArray{<:Real,2}
    outcome_space::OC
    outcomes
    approximation_method::AM
end

#convenience constructor to switch out approximation_method
TransferOperatorApproximation(to, approximation_method) =
    TransferOperatorApproximation(to.transfermatrix, to.outcome_space, to.outcomes, approximation_method)

struct ApproximationIterative <: ApproximationMethod
    N::Int 
    tolerance::Float64 
    delta::Float64
    rng
end

#constructor with default method parameters
#N,tol,delta,rng
pars_default_iterative = (200, 1e-8, 1e-8, Random.default_rng())
ApproximationIterative() = ApproximationIterative(pars_default_iterative...)

struct ApproximationEigen <: ApproximationMethod 
    tol
    krylovdim 
    maxiter
    orth
end

#constructor with default method parameters
ApproximationEigen() = ApproximationEigen(KrylovDefaults.tol,
    KrylovDefaults.krylovdim, 
    KrylovDefaults.maxiter,KrylovDefaults.orth)

"""
    transferoperator(o::OutcomeSpace,x;
        boundary_condition = :none,
        approximation_method=ApproximationIterative()) → to::TransferOperatorApproximation

Approximate the transfer operator given a set of sequentially ordered points (time series) `x` subject to 
an outcome space given by the `o::OutcomeSpace`. 
The keywords `boundary_condition = :none` `boundary_condition = :ApproximationIterative()` are as in [`TransferOperator`](@ref).
"""
function transferoperator(o::OutcomeSpace,x::Array_or_SSSet;
        boundary_condition = :none,
        approximation_method=ApproximationIterative())
    
    #warning (only when used with some kind of binning)
    if typeof(o) <: ValueBinning  && !o.binning.precise
        @warn "`binning.precise == false`. You may be getting points outside the binning."
    end

    outcomes = codify(o,x)
    L = length(outcomes)

    # There are L number of outcomes
    # turn the time series of outcomes into a sequence of unique indices of outcomes
    unique_indices,unique_outcomes = inds_in_terms_of_unique(outcomes, false) # set to true when sorting is fixed
    N = length(unique_outcomes)
   
    #apply boundary conditions (default is :none)
    if boundary_condition == :circular
        append!(unique_indices, [1])
        L += 1
    elseif boundary_condition == :random
        append!(unique_indices, [rand(rng, 1:length(unique_indices))])
        L += 1
    elseif boundary_condition != :none
        error("Boundary condition $(boundary_condition) not implemented")
    end

    #matrix to store the occurrence counts of each transition
	Q = spzeros(N, N)

	#count transitions in Q, assuming symbols from 1 to N
	for i in 1:(L - 1)
        Q[unique_indices[i],unique_indices[i+1]] += 1.0
	end

    #normalize Q (not strictly necessary) and fill P by normalizing rows of Q
    Q .= Q./sum(Q)
    P = normalize_transition_matrix(Q)

    return TransferOperatorApproximation(P, o, unique_outcomes,approximation_method)
end

"""
    InvariantMeasure(to, ρ)

Minimal return struct for [`invariantmeasure`](@ref) that contains the estimated invariant
measure `ρ`, as well as the transfer operator `to` from which it is computed (including
outcome information).
See also: [`invariantmeasure`](@ref).
"""
struct InvariantMeasure{T}
    to::T
    ρ::Probabilities
end

function invariantmeasure(iv::InvariantMeasure)
    return iv.ρ, iv.to.outcomes
end


import LinearAlgebra: norm
"""
## Probabilities and bin information

    invariantmeasure(iv::InvariantMeasure) → (ρ::Probabilities, bins::Vector{<:SVector})

From a pre-computed invariant measure, return the probabilities and associated bins.
The element `ρ[i]` is the probability of visitation to the box `bins[i]`.
See also: [`InvariantMeasure`](@ref).
"""


"""
    invariantmeasure(o::OutcomeSpace, x::Array_or_SSSet; 
        approximation_method=ApproximationIterative()) → iv::InvariantMeasure

Estimate an invariant measure of the approximate transfer operator over the points in `x` based on the provided outcome space and 
approximation method. This is done by first constructing the transfer operator `to` by counting transitions between outcomes using 
`transferoperator`, then calling `invariantmeasure(to::TransferOperatorApproximation)`.
Assumes that the input data are sequential.

Details on the estimation procedure is found the [`transferoperator`](@ref) and [`TransferOperator`](@ref) docstring.

## Example

```julia
using DynamicalSystems
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
orbit, t = trajectory(ds, 20_000; Ttr = 10)

# Estimate the invariant measure over some coarse graining of the orbit.
vb = ValueBinning(RectangularBinning(15))
iv = invariantmeasure(vb,orbit)

# Get the probabilities and the corresponding outcome indexes
ρ,outs = invariantmeasure(iv)
```
"""
function invariantmeasure(o::OutcomeSpace, x::Array_or_SSSet; approximation_method=ApproximationIterative())

    to = transferoperator(o, x; approximation_method=approximation_method) #returns a TransferOperatorApproximation

    return invariantmeasure(to)

end


"""
    invariantmeasure(to::TransferOperatorApproximation{<:OutcomeSpace,ApproximationIterative}) → iv::InvariantMeasure

Return an `Invariantmeasure` containing the invariant measure approximation computed using `ApproximationIterative`.

See also: [`ApproximationIterative`](@ref).
"""
function invariantmeasure(to::TransferOperatorApproximation{<:OutcomeSpace,ApproximationIterative})

    N, tolerance, delta, rng = to.approximation_method.N, to.approximation_method.tolerance, 
        to.approximation_method.delta, to.approximation_method.rng

    P = to.transfermatrix
    #=
    # Start with a random distribution `ρ` (rho). Normalise it so that it
    # sums to 1 and forms a true probability distribution over the partition elements.
    =#
    ρ = rand(rng, Float64, 1, size(P, 1))
    ρ = ρ ./ sum(ρ, dims = 2)
    
    #=
    # Start estimating the invariant distribution. We could either do this by
    # finding the left-eigenvector of M, or by repeated application of M on Ρ
    # until the distribution converges. Here, we use the latter approach,
    # meaning that we iterate until Ρ doesn't change substantially between
    # iterations.
    =#
    distribution = ρ * P
    distance = norm(distribution - ρ) / norm(ρ)

    check = floor(Int, 1 / delta)
    check_pts = floor.(Int, transpose(collect(1:N)) ./ check) .* transpose(collect(1:N))
    check_pts = check_pts[check_pts .> 0]
    num_checkpts = size(check_pts, 1)
    check_pts_counter = 1

    counter = 1
    while counter <= N && distance >= tolerance
        counter += 1
        ρ = distribution

        # Apply the Markov matrix to the current state of the distribution
        distribution = ρ * P

        if (check_pts_counter <= num_checkpts &&
           counter == check_pts[check_pts_counter])

            check_pts_counter += 1
            colsum_distribution = sum(distribution, dims = 2)[1]
            if abs(colsum_distribution - 1) > delta
                distribution = distribution ./ colsum_distribution
            end
        end
        distance = norm(distribution - ρ) / norm(ρ)
    end
    distribution = dropdims(distribution, dims = 1)

    # Do the last normalisation and check
    colsum_distribution = sum(distribution)

    if abs(colsum_distribution - 1) > delta
        distribution = distribution ./ colsum_distribution
    end
    # Extract the elements of the invariant measure corresponding to these indices
    return InvariantMeasure(to, Probabilities(distribution))
end

"""
    invariantmeasure(to::TransferOperatorApproximation{<:OutcomeSpace,ApproximationEigen}) → iv::InvariantMeasure

Return an `Invariantmeasure` containing the invariant measure approximation computed using `ApproximationEigen`.

See also: [`ApproximationEigen`](@ref).
"""
function invariantmeasure(to::TransferOperatorApproximation{<:OutcomeSpace,ApproximationEigen})
    P = to.transfermatrix
    #first eigenvalue with Largest Real part
    vals, vecs, info = eigsolve(P', 1, :LR)
    info.converged < 1 && @warn "KrylovKit.eigsolve did not converge!"
    ρ = real.(vecs[1]) ./ sum(real.(vecs[1]))
    return InvariantMeasure(to, Probabilities(ρ.nzval))
end

"""
    transfermatrix(iv::InvariantMeasure) → M::AbstractArray{<:Real, 2}

Return the transfer matrix/operator. Thus, the entry `M[i, j]` is the
probability of jumping from the state `i` to the state `j`.

See also: [`TransferOperator`](@ref).
"""
function transfermatrix(iv::InvariantMeasure)
    return iv.to.transfermatrix
end


# Explicitly extend `probabilities` because we can skip the decoding step, which is 
# expensive.
function probabilities(probest::TransferOperator, o::OutcomeSpace, x::Array_or_SSSet)
    verify_counting_based(o, "TransferOperator")

    approx_method = probest.approximation_method
    boundary_cond = probest.boundary_condition
    to = transferoperator(o, x; approximation_method=approx_method, boundary_condition=boundary_cond)
    outs = to.outcomes
    ρ =  invariantmeasure(to).ρ

    #if o isa ValueBinning, return bins in order
    if o isa ValueBinning
        ordering = sortperm(outs)
        outs_ordered = outs[ordering]
        return Probabilities(ρ.p[ordering], (outs_ordered[ordering],))
    end

    outs_decoded = [decode(o.encoding, oc) for oc in outs] # outcomes decoded from integers
    return Probabilities(ρ, (outs_decoded,))
end

function probabilities_and_outcomes(probest::TransferOperator, o::OutcomeSpace, x::Array_or_SSSet)
    verify_counting_based(o, "TransferOperator")


    approx_method = probest.approximation_method
    boundary_cond = probest.boundary_condition
    to = transferoperator(o, x; approximation_method=approx_method, boundary_condition=boundary_cond)
    outs = to.outcomes
    ρ = invariantmeasure(to).ρ

    #different for ValueBinning outcome space 
    if o isa ValueBinning
        ordering = sortperm(outs) #get bins in the correct order
        outs_decoded = [decode(RectangularBinEncoding(o.binning, x), i) for i in outs] #include decode step here
        return Probabilities(ρ.p[ordering], (outs_decoded[ordering],)), outs_decoded[ordering]
    end
    
    #for other outcome spaces use decoding
    outs_decoded = [decode(o.encoding, oc) for oc in outs] # outcomes decoded from integers

    probs = Probabilities(ρ, (outs_decoded,))

    return probs, outs_decoded
end

