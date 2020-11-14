using GroupSlices, DelayEmbeddings, SparseArrays

export TransferOperator, 
    transferoperator, TransferOperatorApproximationRectangular,
    InvariantMeasureEstimate, invariantmeasure,
    binhist

"""
    TransferOperator(r::RectangularBinning)

A probability estimator based on binning data into rectangular boxes dictated by 
the binning scheme `r`, then estimating the transfer (Perron-Frobenius) operator 
over the bins. Assumes that the input data are sequential.

The transfer operator ``P^{N}``is computed as an `N`-by-`N` matrix of transition 
probabilities between the states defined by the partition elements, where `N` is the 
number of boxes in the partition that is visited by the orbit/points. The left invariant
distribution ``\\rho^{N}``, where ``\\rho^{N}P^{N} = \\rho^{N}``, approximates the 
invariant density of the system subject to the partition `r`. Thus, ``\\rho^{N}`` can 
be taken as a probability distribution over the partition elements.

This implementation follows the approach in Diego et al. (2019)[^Diego2019].

See also: [`RectangularBinning`](@ref).

[^Diego2019]: Diego, D., Haaga, K. A., & Hannisdal, B. (2019). Transfer entropy computation using the Perron-Frobenius operator. Physical Review E, 99(4), 042212.
"""
struct TransferOperator{R} <: BinningProbabilitiesEstimator
    binning::R
    
    function TransferOperator(binning::R) where R <: RectangularBinning
        new{R}(binning)
    end
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

inds_in_terms_of_unique(x::Dataset) = inds_in_terms_of_unique(x.data)

"""
    TransferOperatorApproximationRectangular(to, ϵ::RectangularBinning, mini, edgelengths, bins)

The `N`-by-`N` matrix `to` is an approximation to the transfer operator, subject to the 
partition `ϵ`, computed over some set of sequentially ordered points. 

For convenience, `mini` and `edgelengths` provide the minima and box edge lengths along 
each coordinate axis, as determined by applying `ϵ` to the points. The coordinates of 
the (leftmost, if axis is ordered low-high) box corners are given in `bins`. 

Only bins actually visited by the points are considered, and `bins` give the coordinates 
of these bins. The element `bins[i]` correspond to the `i`-th state of the system, which 
corresponds to the `i`-th column/row of the transfer operator `to`.

See also: [`RectangularBinning`](@ref).
"""
struct TransferOperatorApproximationRectangular
    to
    ϵ::RectangularBinning
    mini
    edgelengths
    bins
end

"""
    transferoperator(pts::AbstractDataset{D, T}, ϵ::RectangularBinning) → TransferOperatorApproximationRectangular

Estimate the transfer operator given a set of sequentially ordered points subject to a 
rectangular partition given by `ϵ`.

## Example 

```julia
using DynamicalSystems, Entropy
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
    mini, edgelengths = Entropies.minima_edgelengths(pts, ϵ)

    # The L points visits a total of L bins, which are the following bins: 
    visited_bins = Entropies.encode_as_bin(pts, mini, edgelengths)
    sort!(visited_bins)
    
    # There are N=length(unique(visited_bins)) unique bins.
    # Which of the unqiue bins does each of the L points visit? 
    visits_whichbin = inds_in_terms_of_unique(visited_bins, true)

    # `visitors` lists the indices of the points visiting each of the N unique bins.
    slices = groupslices(visited_bins)
    visitors = groupinds(slices) 
    
    # first_visited_by == [x[1] for x in visitors]
    first_visited_by = firstinds(slices)
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
    bins = [β .* edgelengths .+ mini for β in visited_bins]

    TransferOperatorApproximationRectangular(TO, ϵ, mini, edgelengths, bins)
end

""" 
    InvariantMeasureEstimate(to, ρ::Probabilities)

The estimated invariant measure `ρ` associated with some transfer operator `to`.
""" 
struct InvariantMeasureEstimate{T}
    to::T
    ρ::Probabilities

    function InvariantMeasureEstimate(
        to::T, ρ::Probabilities) where T<:TransferOperatorApproximationRectangular
        new{T}(to, ρ)
    end
end


import LinearAlgebra: norm
"""
    invariantmeasure(to::TransferOperatorApproximationRectangular) → InvariantMeasureEstimate

Estimate the invariant measure associated with some pre-computed transfer operator.

## Example 

```julia
using DynamicalSystems, Entropy
D = 4
ds = Systems.lorenz96(D; F = 32.0)
N, dt = 20000, 0.1
orbit = trajectory(ds, N*dt; dt = dt, Ttr = 10.0)

# Estimate transfer operator over some coarse graining of the orbit.
to = transferoperator(orbit, RectangularBinning(10))
invariantmeasure(to)
```

See also: [`TransferOperatorApproximationRectangular`](@ref).
"""
function invariantmeasure(to::TransferOperatorApproximationRectangular; 
        N::Int = 200, tolerance::Float64 = 1e-8, delta::Float64 = 1e-8)
    
    TO = to.to
    #=
    # Start with a random distribution `Ρ` (big rho). Normalise it so that it
    # sums to 1 and forms a true probability distribution over the partition elements.
    =#
    Ρ = rand(Float64, 1, size(TO, 1))
    Ρ = Ρ ./ sum(Ρ, dims = 2)

    #=
    # Start estimating the invariant distribution. We could either do this by
    # finding the left-eigenvector of M, or by repeated application of M on Ρ
    # until the distribution converges. Here, we use the latter approach,
    # meaning that we iterate until Ρ doesn't change substantially between
    # iterations.
    =#
    distribution = Ρ * TO

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
        distribution = Ρ * TO

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
    δ = tolerance/size(TO, 1)
    inds_nonzero = findall(distribution .> δ)

    # Extract the elements of the invariant measure corresponding to these indices
    return InvariantMeasureEstimate(to, Probabilities(distribution))
end


function probabilities(x::AbstractDataset, est::TransferOperator{RectangularBinning})
    to = transferoperator(x, est.binning)
    iv = invariantmeasure(to)

    return iv.ρ
end


function binhist(x::AbstractDataset, est::TransferOperator{RectangularBinning})
    to = transferoperator(x, est.binning)
    iv = invariantmeasure(to)

    return iv.ρ, to.bins
end

"""
    binhist(iv::InvariantMeasureEstimate) → ρ, bins
    binhist(to::TransferOperator{RectangularBinning}) → ρ, bins
    binhist(x::AbstractDataset, est::TransferOperator{RectangularBinning})→ ρ, bins 

Analogous to `binhist` for a rectangular binning, but here probabilities are estimated from 
an approximation to the transfer operator subject to that binning.

See also: [`TransferOperatorApproximationRectangular`](@ref), 
[`InvariantMeasureEstimate`](@ref), [`TransferOperator`](@ref).
"""
function binhist(iv::InvariantMeasureEstimate)
    return iv.ρ, iv.to.bins
end

function binhist(to::TransferOperatorApproximationRectangular)
    iv = invariantmeasure(to)
    return iv.ρ, to.bins
end