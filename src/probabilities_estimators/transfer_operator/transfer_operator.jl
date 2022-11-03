# TODO: This needs its own docpage I feel.
using DelayEmbeddings, SparseArrays
using StaticArrays

include("GroupSlices.jl")

export
    TransferOperator, # the probabilities estimator
    InvariantMeasure, invariantmeasure,
    transfermatrix

"""
    TransferOperator(ϵ::RectangularBinning) <: ProbabilitiesEstimator

A probability estimator based on binning data into rectangular boxes dictated by
the binning scheme `ϵ`, then approxmating the transfer (Perron-Frobenius) operator
over the bins, then taking the invariant measure associated with that transfer operator
as the bin probabilities. Assumes that the input data are sequential (time-ordered).

This implementation follows the grid estimator approach in Diego et al. (2019)[^Diego2019].

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
{\\#\\{ x_n | \\phi(x_n) \\in C_j \\cap x_n \\in C_i \\}}
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
``\\mathbf{\\rho}^N`` approximates the invariant density of the system subject to the
partition `ϵ`, and can be taken as a probability distribution over the partition elements.

In practice, the invariant measure ``\\mathbf{\\rho}^N`` is computed using
[`invariantmeasure`](@ref), which also approximates the transfer matrix. The invariant distribution
is initialized as a length-`N` random distribution which is then applied to ``P^{N}``.
The resulting length-`N` distribution is then applied to ``P^{N}`` again. This process
repeats until the difference between the distributions over consecutive iterations is
below some threshold.

## Probability and entropy estimation

- `probabilities(x::AbstractDataset, est::TransferOperator{RectangularBinning})` estimates
    probabilities for the bins defined by the provided binning (`est.ϵ`)
- `entropy_renyi(x::AbstractDataset, est::TransferOperator{RectangularBinning})` does the same,
    but computes generalized entropy using the probabilities.


See also: [`RectangularBinning`](@ref), [`invariantmeasure`](@ref).

[^Diego2019]: Diego, D., Haaga, K. A., & Hannisdal, B. (2019). Transfer entropy computation using the Perron-Frobenius operator. Physical Review E, 99(4), 042212.
"""
struct TransferOperator{R} <: ProbabilitiesEstimator
    ϵ::R
end

# If x is not sorted, we need to look at all pairwise comparisons
function inds_in_terms_of_unique(x)
    U = unique(x)
    N = length(x)
    Nu = length(U)
    inds = zeros(Int, N)

    for j = 1:N
        xⱼ = view(x, j)
        for i = 1:Nu
            # using views doesn't allocate
            @inbounds if xⱼ == view(U, i)
                inds[j] = i
            end
        end
    end

    return inds
end

# Taking advantage of the fact that x is sorted reduces runtime by 1.5 orders of magnitude
# for datasets of >100 000+ points
function inds_in_terms_of_unique_sorted(x) # assumes sorted
    @assert issorted(x)
    U = unique(x)
    N, Nu = length(x), length(U)
    prev = view(x, 1)
    inds = zeros(Int, N)
    uidx = 1
    @inbounds for j = 1:N
        xⱼ = view(x, j)
        # if the current value has changed, then we know that the corresponding index
        # for the unique point must be incremented by 1
        if xⱼ != prev
            prev = xⱼ
            uidx += 1
        end
        inds[j] = uidx
    end

    return inds
end

function inds_in_terms_of_unique(x, sorted::Bool)
    if sorted
        return inds_in_terms_of_unique_sorted(x)
    else
        return inds_in_terms_of_unique(x)
    end
end

inds_in_terms_of_unique(x::AbstractDataset) = inds_in_terms_of_unique(x.data)


"""
    TransferOperatorApproximationRectangular(to, ϵ::RectangularBinning, mini, edgelengths,
        bins, sort_idxs)

The `N`-by-`N` matrix `to` is an approximation to the transfer operator, subject to the
partition `ϵ`, computed over some set of sequentially ordered points.

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
struct TransferOperatorApproximationRectangular{T<:Real}
    transfermatrix::AbstractArray{T, 2}
    ϵ::RectangularBinning
    mini
    edgelengths
    bins
    sort_idxs::Vector{Int}
    visitors::Vector{Vector{Int}}
end
# TODO: The above is type unstable!

"""
    transferoperator(pts::AbstractDataset{D, T}, ϵ::RectangularBinning) → TransferOperatorApproximationRectangular

Estimate the transfer operator given a set of sequentially ordered points subject to a
rectangular partition given by `ϵ`.

## Example

```julia
using DynamicalSystems, Plots, Entropy
D = 4
ds = Systems.lorenz96(D; F = 32.0)
N, dt = 20000, 0.1
orbit = trajectory(ds, N*dt; dt = dt, Ttr = 10.0)

# Estimate transfer operator over some coarse graining of the orbit.
transferoperator(orbit, RectangularBinning(10))
```

See also: [`RectangularBinning`](@ref).
"""
function transferoperator(pts::AbstractDataset{D, T}, ϵ::RectangularBinning;
        boundary_condition = :circular) where {D, T<:Real}

    L = length(pts)
    encoder = RectangularBinEncoding(pts, ϵ)

    # The L points visits a total of L bins, which are the following bins:
    visited_bins = outcomes(pts, encoder)
    sort_idxs = sortperm(visited_bins)

    # TODO: fix re-indexing after sorting. Sorting is much faster, so we want to do so.
    #sort!(visited_bins)

    # There are N=length(unique(visited_bins)) unique bins.
    # Which of the unqiue bins does each of the L points visit?
    visits_whichbin = inds_in_terms_of_unique(visited_bins, false) # set to true when sorting is fixed

    # `visitors` lists the indices of the points visiting each of the N unique bins.
    slices = GroupSlices.groupslices(visited_bins)
    visitors = GroupSlices.groupinds(slices)

    # first_visited_by == [x[1] for x in visitors]
    first_visited_by = GroupSlices.firstinds(slices)
    L = length(first_visited_by)

    I = Int32[]
    J = Int32[]
    P = Float64[]

    # Preallocate target index for the case where there is only
    # one point of the orbit visiting a bin.
    target_bin_j::Int = 0
    n_visitsᵢ::Int = 0

    if boundary_condition == :circular
        #warn("Using circular boundary condition")
        append!(visits_whichbin, [1])
    elseif boundary_condition == :random
        #warn("Using random circular boundary condition")
        append!(visits_whichbin, [rand(1:length(visits_whichbin))])
    else
        error("Boundary condition $(boundary_condition) not implemented")
    end

    # Loop over the visited bins bᵢ
    for i in 1:L
        # How many times is this bin visited?
        n_visitsᵢ = length(visitors[i])

        # If both conditions below are true, then there is just one
        # point visiting the i-th bin. If there is only one visiting point and
        # it happens to be the last, we skip it, because we don't know its
        # image.
        if n_visitsᵢ == 1 && !(i == visits_whichbin[end])
            # To which bin does the single point visiting bᵢ jump if we
            # shift it one time step ahead along its orbit?
            target_bin_j = visits_whichbin[visitors[i][1] + 1][1]

            # We now know that exactly one point (the i-th) does the
            # transition from i to the target j.
            push!(I, i)
            push!(J, target_bin_j)
            push!(P, 1.0)
        end

        # If more than one point of the orbit visits the i-th bin, we
        # identify the visiting points and track which bins bⱼ they end up
        # in after the forward linear map of the points.
        if n_visitsᵢ > 1
            timeindices_visiting_pts = visitors[i]

            # TODO: Introduce circular boundary condition. Simply excluding
            # might lead to a cascade of loosing points.

            # If bᵢ is the bin visited by the last point in the orbit, then
            # the last entry of `visiting_pts` will be the time index of the
            # last point of the orbit. In the next time step, that point will
            # have nowhere to go along its orbit (precisely because it is the
            # last data point). Thus, we exclude it.
            if i == visits_whichbin[end]
                #warn("Removing last point")
                n_visitsᵢ = length(timeindices_visiting_pts) - 1
                timeindices_visiting_pts = timeindices_visiting_pts[1:(end - 1)]
            end

            # To which boxes do each of the visitors to bᵢ jump in the next
            # time step?
            target_bins = visits_whichbin[timeindices_visiting_pts .+ 1]
            unique_target_bins = unique(target_bins)

            # Count how many points jump from the i-th bin to each of
            # the unique target bins, and use that to calculate the transition
            # probability from bᵢ to bⱼ.
            #for j in 1:length(unique_target_bins)
            for (j, bᵤ) in enumerate(unique(target_bins))
                n_transitions_i_to_j = sum(target_bins .== bᵤ)

                push!(I, i)
                push!(J, bᵤ)
                push!(P, n_transitions_i_to_j / n_visitsᵢ)
            end
        end
    end

    # Transfer operator is just the normalized transition probabilities between the boxes.
    TO = sparse(I, J, P)

    # Compute the coordinates of the visited bins. bins[i] corresponds to the i-th
    # row/column of the transfer operator
    unique!(visited_bins)
    bins = [β .* encoder.edgelengths .+ encoder.mini for β in visited_bins]

    TransferOperatorApproximationRectangular(TO, ϵ, encoder.mini, encoder.edgelengths, bins,
        sort_idxs, visitors)
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
    invariantmeasure(x::AbstractDataset, ϵ::RectangularBinning) → iv::InvariantMeasure

Estimate an invariant measure over the points in `x` based on binning the data into
rectangular boxes dictated by the binning scheme `ϵ`, then approximate the transfer
(Perron-Frobenius) operator over the bins. From the approximation to the transfer operator,
compute an invariant distribution over the bins. Assumes that the input data are sequential.

Details on the estimation procedure is found the [`TransferOperator`](@ref) docstring.

## Example

```julia
using DynamicalSystems, Plots, Entropies
D = 4
ds = Systems.lorenz96(D; F = 32.0)
N, dt = 20000, 0.1
orbit = trajectory(ds, N*dt; dt = dt, Ttr = 10.0)

# Estimate the invariant measure over some coarse graining of the orbit.
iv = invariantmeasure(orbit, RectangularBinning(15))

# Get the probabilities and bins
invariantmeasure(iv)
```

## Probabilities and bin information

    invariantmeasure(iv::InvariantMeasure) → (ρ::Probabilities, bins::Vector{<:SVector})

From a pre-computed invariant measure, return the probabilities and associated bins.
The element `ρ[i]` is the probability of visitation to the box `bins[i]`. Analogous to
[`binhist`](@ref).


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
        N::Int = 200, tolerance::Float64 = 1e-8, delta::Float64 = 1e-8)

    TO = to.transfermatrix
    #=
    # Start with a random distribution `Ρ` (big rho). Normalise it so that it
    # sums to 1 and forms a true probability distribution over the partition elements.
    =#
    Ρ = rand(Float64, 1, size(to.transfermatrix, 1))
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

    # Find partition elements with strictly positive measure.
    δ = tolerance/size(to.transfermatrix, 1)
    inds_nonzero = findall(distribution .> δ)

    # Extract the elements of the invariant measure corresponding to these indices
    return InvariantMeasure(to, Probabilities(distribution))
end

function invariantmeasure(x::AbstractDataset, ϵ::RectangularBinning)
    to = transferoperator(x, ϵ)
    invariantmeasure(to)
end

function probabilities(x::AbstractDataset, est::TransferOperator)
    to = transferoperator(x, est.ϵ)
    iv = invariantmeasure(to)
    return iv.ρ
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

"""
    entropy_transferoperator(x, binning::RectangularBinning; base = MathConstants.e)

Compute the (Shannon) entropy of `x` using an approximation to the transfer operator over
the state-space coarse graining produced by the provided `binning` scheme.

Short-hand for `renyi_entropy(x, TransferOperator(binning); base = base, q = 1)`.

See also: [`TransferOperator`](@ref).

[^Diego2019]: Diego, D., Haaga, K. A., & Hannisdal, B. (2019). Transfer entropy computation
    using the Perron-Frobenius operator. Physical Review E, 99(4), 042212.
"""
function entropy_transferoperator(x, b::RectangularBinning)
    est = TransferOperator(binning)
    renyi_entropy(x, est; base = base, q = 1)
end
