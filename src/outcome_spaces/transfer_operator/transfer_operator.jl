using DelayEmbeddings, SparseArrays
using StaticArrays
using Random

include("utils.jl")

export
    TransferOperator, # the probabilities estimator
    InvariantMeasure, invariantmeasure,
    transfermatrix, transferoperator

"""
    TransferOperator <: OutcomeSpace
    TransferOperator(b::AbstractBinning; warn_precise = true, rng = Random.default_rng())

An [`OutcomeSpace`](@ref) based on binning data into rectangular boxes dictated by
the given binning scheme `b`.

When used with [`probabilities`](@ref), then the transfer (Perron-Frobenius) operator
is approximated over the bins, then bin probabilities are estimated as the invariant measure
associated with that transfer operator. Assumes that the input data are sequential
(time-ordered).

This implementation follows the grid estimator approach in [Diego2019](@citet).

## Precision

The default behaviour when using [`RectangularBinning`](@ref) or
[`FixedRectangularBinning`](@ref) is to accept some loss of precision on the 
bin boundaries for speed-ups, but this may lead to issues for `TransferOperator`
where some points may be encoded as the symbol `-1` ("outside the binning").
The `warn_precise` keyword controls whether the user is warned when a less 
precise binning is used.

## Outcome space

The outcome space for `TransferOperator` is the set of unique bins constructed
from `b`. Bins are identified by their left (lowest-value) corners, are given in
data units, and are returned as `SVector`s.

## Bin ordering

Bins returned by [`probabilities_and_outcomes`](@ref) are ordered according to first
appearance (i.e. the first time the input (multivariate) timeseries visits the bin).
Thus, if

```julia
b = RectangularBinning(4)
est = TransferOperator(b)
probs, outcomes = probabilities_and_outcomes(x, est) # x is some timeseries
```

then `probs[i]` is the invariant measure (probability) of the bin `outcomes[i]`, which is
the `i`-th bin visited by the timeseries with nonzero measure.

## Description

The transfer operator ``P^{N}``is computed as an `N`-by-`N` matrix of transition
probabilities between the states defined by the partition elements, where `N` is the
number of boxes in the partition that is visited by the orbit/points.

If  ``\\{x_t^{(D)} \\}_{n=1}^L`` are the ``L`` different ``D``-dimensional points over
which the transfer operator is approximated, ``\\{ C_{k=1}^N \\}`` are the ``N`` different
partition elements (as dictated by `ϵ`) that gets visited by the points, and
 ``\\phi(x_t) = x_{t+1}``, then

```math
P_{ij} = \\dfrac
{\\#\\{ x_n | \\phi(x_n) \\in C_j \\cap x_n \\in C_i \\}}
{\\#\\{ x_m | x_m \\in C_i \\}},
```

where ``\\#`` denotes the cardinal. The element ``P_{ij}`` thus indicates how many points
that are initially in box ``C_i`` end up in box ``C_j`` when the points in ``C_i`` are
projected one step forward in time. Thus, the row ``P_{ik}^N`` where
``k \\in \\{1, 2, \\ldots, N \\}`` gives the probability
of jumping from the state defined by box ``C_i`` to any of the other ``N`` states. It
follows that ``\\sum_{k=1}^{N} P_{ik} = 1`` for all ``i``. Thus, ``P^N`` is a row/right
stochastic matrix.

### Invariant measure estimation from transfer operator

The left invariant distribution ``\\mathbf{\\rho}^N`` is a row vector, where
``\\mathbf{\\rho}^N P^{N} = \\mathbf{\\rho}^N``. Hence, ``\\mathbf{\\rho}^N`` is a row
eigenvector of the transfer matrix ``P^{N}`` associated with eigenvalue 1. The distribution
``\\mathbf{\\rho}^N`` approximates the invariant density of the system subject to
`binning`, and can be taken as a probability distribution over the partition elements.

In practice, the invariant measure ``\\mathbf{\\rho}^N`` is computed using
[`invariantmeasure`](@ref), which also approximates the transfer matrix. The invariant
distribution is initialized as a length-`N` random distribution which is then applied to
``P^{N}``. For reproducibility in this step, set the `rng`.
The resulting length-`N` distribution is then applied to ``P^{N}`` again. This process
repeats until the difference between the distributions over consecutive iterations is
below some threshold.

See also: [`RectangularBinning`](@ref), [`FixedRectangularBinning`](@ref),
[`invariantmeasure`](@ref).
"""
struct TransferOperator{R<:AbstractBinning, RNG} <: OutcomeSpace
    binning::R
    warn_precise::Bool
    rng::RNG
    function TransferOperator(b::R; 
            rng::RNG = Random.default_rng(), 
            warn_precise = true) where {R <: AbstractBinning, RNG}
        return new{R, RNG}(b, warn_precise, rng)
    end
end
function TransferOperator(ϵ::Union{Real,Vector}; 
        rng = Random.default_rng(), warn_precise = true)
    return TransferOperator(RectangularBinning(ϵ), warn_precise, rng)
end



"""
    TransferOperatorApproximationRectangular(to, binning::RectangularBinning, mini,
        edgelengths, bins, sort_idxs)

The `N`-by-`N` matrix `to` is an approximation to the transfer operator, subject to the
given `binning`, computed over some set of sequentially ordered points.

For convenience, `mini` and `edgelengths` provide the minima and box edge lengths along
each coordinate axis, as determined by applying `ϵ` to the points. The coordinates of
the (leftmost, if axis is ordered low-high) box corners are given in `bins`.

Only bins actually visited by the points are considered, and `bins` give the coordinates
of these bins. The element `bins[i]` correspond to the `i`-th state of the system, which
corresponds to the `i`-th column/row of the transfer operator `to`.

`sort_idxs` contains the indices that would sort the input points. `visitors` is a
vector of vectors, where `visitors[i]` contains the indices of the (sorted)
points that visits `bins[i]`.

See also: [`RectangularBinning`](@ref).
"""
struct TransferOperatorApproximationRectangular{
        T<:Real,
        BINS,
        E}
    transfermatrix::AbstractArray{T, 2}
    encoding::E
    bins::BINS
end

"""
    transferoperator(pts::AbstractStateSpaceSet,
        binning::Union{FixedRectangularBinning, RectangularBinning};boundary_condition = :none,warn_precise = true) → TransferOperatorApproximationRectangular

Estimate the transfer operator given a set of sequentially ordered points subject to a
rectangular partition given by the `binning`.
"""
function transferoperator(pts::AbstractStateSpaceSet{D, T},
        binning::Union{FixedRectangularBinning, RectangularBinning};
        boundary_condition = :none, 
        warn_precise = true) where {D, T<:Real}

    L = length(pts)
    if warn_precise && !binning.precise
        @warn "`binning.precise == false`. You may be getting points outside the binning."
    end
    encoding = RectangularBinEncoding(binning, pts)

    # The L points visits a total of N bins, which are the following bins (referenced
    # here as cartesian coordinates, not absolute bins):
    outcomes = map(pᵢ -> encode(encoding, pᵢ), pts)
    #sort_idxs = sortperm(visited_bins)
    #sort!(visited_bins) # see todo on github

    # There are N=length(unique(visited_bins)) unique bins.
    # Which of the unqiue bins does each of the L points visit?
    visits_whichbin,unique_outcomes = inds_in_terms_of_unique(outcomes, false) # set to true when sorting is fixed
    N = length(unique_outcomes)
   
    #apply boundary conditions (default is :none)
    if boundary_condition == :circular
        append!(visits_whichbin, [1])
        L += 1
    elseif boundary_condition == :random
        append!(visits_whichbin, [rand(rng, 1:length(visits_whichbin))])
        L += 1
    elseif boundary_condition != :none
        error("Boundary condition $(boundary_condition) not implemented")
    end

    #matrix to store the occurrence counts of each transition
	Q = spzeros(N, N)

	#count transitions in Q, assuming symbols from 1 to N
	for i in 1:(L - 1)
        Q[visits_whichbin[i],visits_whichbin[i+1]] += 1.0
	end

    #normalize Q (not strictly necessary) and fill P by normalizing rows of Q
    Q .= Q./sum(Q)
    P = calculate_transition_matrix(Q)

    unique!(outcomes)
    return TransferOperatorApproximationRectangular(
        P, encoding, outcomes)
end

"""
    InvariantMeasure(to, ρ)

Minimal return struct for [`invariantmeasure`](@ref) that contains the estimated invariant
measure `ρ`, as well as the transfer operator `to` from which it is computed (including
bin information).

See also: [`invariantmeasure`](@ref).
"""
struct InvariantMeasure{T}
    to::T
    ρ::Probabilities
end

function invariantmeasure(iv::InvariantMeasure)
    return iv.ρ, iv.to.bins
end


import LinearAlgebra: norm

"""
    invariantmeasure(x::AbstractStateSpaceSet, binning::RectangularBinning;
        rng = Random.default_rng()) → iv::InvariantMeasure

Estimate an invariant measure over the points in `x` based on binning the data into
rectangular boxes dictated by the `binning`, then approximate the transfer
(Perron-Frobenius) operator over the bins. From the approximation to the transfer operator,
compute an invariant distribution over the bins. Assumes that the input data are sequential.

Details on the estimation procedure is found the [`TransferOperator`](@ref) docstring.

## Example

```julia
using DynamicalSystems
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
orbit, t = trajectory(ds, 20_000; Ttr = 10)

# Estimate the invariant measure over some coarse graining of the orbit.
iv = invariantmeasure(orbit, RectangularBinning(15))

# Get the probabilities and bins
invariantmeasure(iv)
```

## Probabilities and bin information

    invariantmeasure(iv::InvariantMeasure) → (ρ::Probabilities, bins::Vector{<:SVector})

From a pre-computed invariant measure, return the probabilities and associated bins.
The element `ρ[i]` is the probability of visitation to the box `bins[i]`.


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

See also: [`InvariantMeasure`](@ref).
"""
function invariantmeasure(to::TransferOperatorApproximationRectangular;
        N::Int = 200, tolerance::Float64 = 1e-8, delta::Float64 = 1e-8,
        rng = Random.default_rng())

    TO = to.transfermatrix
    #=
    # Start with a random distribution `Ρ` (big rho). Normalise it so that it
    # sums to 1 and forms a true probability distribution over the partition elements.
    =#
    Ρ = rand(rng, Float64, 1, size(to.transfermatrix, 1))
    Ρ = Ρ ./ sum(Ρ, dims = 2)

    #=
    # Start estimating the invariant distribution. We could either do this by
    # finding the left-eigenvector of M, or by repeated application of M on Ρ
    # until the distribution converges. Here, we use the latter approach,
    # meaning that we iterate until Ρ doesn't change substantially between
    # iterations.
    =#
    distribution = Ρ * to.transfermatrix

    distance = norm(distribution - Ρ) / norm(Ρ)

    check = floor(Int, 1 / delta)
    check_pts = floor.(Int, transpose(collect(1:N)) ./ check) .* transpose(collect(1:N))
    check_pts = check_pts[check_pts .> 0]
    num_checkpts = size(check_pts, 1)
    check_pts_counter = 1

    counter = 1
    while counter <= N && distance >= tolerance
        counter += 1
        Ρ = distribution

        # Apply the Markov matrix to the current state of the distribution
        distribution = Ρ * to.transfermatrix

        if (check_pts_counter <= num_checkpts &&
           counter == check_pts[check_pts_counter])

            check_pts_counter += 1
            colsum_distribution = sum(distribution, dims = 2)[1]
            if abs(colsum_distribution - 1) > delta
                distribution = distribution ./ colsum_distribution
            end
        end

        distance = norm(distribution - Ρ) / norm(Ρ)
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

function invariantmeasure(x::AbstractStateSpaceSet,
        binning::Union{FixedRectangularBinning, RectangularBinning};
        warn_precise = true, rng = Random.default_rng())
    to = transferoperator(x, binning; warn_precise)
    return invariantmeasure(to; rng)
end

"""
    transfermatrix(iv::InvariantMeasure) → (M::AbstractArray{<:Real, 2}, bins::Vector{<:SVector})

Return the transfer matrix/operator and corresponding bins. Here, `bins[i]` corresponds
to the i-th row/column of the transfer matrix. Thus, the entry `M[i, j]` is the
probability of jumping from the state defined by `bins[i]` to the state defined by
`bins[j]`.

See also: [`TransferOperator`](@ref).
"""
function transfermatrix(iv::InvariantMeasure)
    return iv.to.transfermatrix, iv.to.bins
end

# Explicitly extend `probabilities` because we can skip the decoding step, which is 
# expensive.
function probabilities(est::TransferOperator, x::Array_or_SSSet)
    to = transferoperator(StateSpaceSet(x), est.binning; 
        warn_precise = est.warn_precise)
    return Probabilities(invariantmeasure(to; rng = est.rng).ρ)
end

function probabilities_and_outcomes(est::TransferOperator, x::Array_or_SSSet)
    to = transferoperator(StateSpaceSet(x), est.binning; 
        warn_precise = est.warn_precise)
    probs = invariantmeasure(to; rng = est.rng).ρ

    # Note: bins are *not* sorted. They occur in the order of first appearance, according
    # to the input time series. Taking the unique bins preserves the order of first
    # appearance.
    bins = to.bins
    unique!(bins)
    outs = decode.(Ref(to.encoding), bins) # coordinates of the visited bins
    probs = Probabilities(probs, (outs, ))
    return probs, outcomes(probs)
end

outcome_space(est::TransferOperator, x) = outcome_space(est.binning, x)
