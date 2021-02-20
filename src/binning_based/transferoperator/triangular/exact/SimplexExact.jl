export SimplexExact, invariantmeasure
import Entropies.invariantmeasure



"""
    SimplexExact

A transfer operator estimator using a triangulation partition and exact 
simplex intersections[^Diego2019]. 

To use this estimator, the Simplices.jl package must be brought into scope by doing 
`using Simplices` after running `using Entropies`. 

*Note: due to computing exact 
simplex intersections, this estimator is slow compared to [`SimplexPoint`](@ref).*

[^Diego2019]: Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
struct SimplexExact
    bc::String
    
    function SimplexExact(bc::String = "circular")
        isboundarycondition(bc, "triangulation")  || error("Boundary condition '$bc' not valid.")
        new(bc)
    end
end
Base.show(io::IO, se::SimplexExact) = print(io, "SimplexExact{$(se.bc)}")


""" Generate a transopergenerator for an exact simplex estimator."""
function transopergenerator(pts, method::TransferOperator{<:SimplexExact})
    # modified points, where the image of each point is guaranteed to lie within the convex hull of the previous points
    invariant_pts = invariantize(pts)
        
    # triangulation of the invariant points. The last point is excluded, so that 
    # the last vertex also can be mapped forward one step in time.
    triang = DelaunayTriangulation(invariant_pts[1:end-1])

    init = (invariant_pts = invariant_pts, triang = triang)

    TransferOperatorGenerator(method, pts, init)
end

function (tog::TransferOperatorGenerator{T})(; tol = 1e-8) where T <: TransferOperator{SimplexExact}
    invariant_pts, triang = getfield.(Ref(tog.init), (:invariant_pts, :triang))
    
    D = length(invariant_pts[1])
    N = length(triang)
    ϵ = tol / N
    
    # Pre-allocate simplex and its image
    image_simplex = MutableSimplex(zeros(Float64, D, D+1))
    simplex = MutableSimplex(zeros(Float64, D, D+1))
    M = zeros(Float64, N, N)
    
    for j in 1:N
        for k = 1:(D + 1)
            # Get the vertices of the image simplex
            image_simplex[k] .= invariant_pts[triang[j][k] + 1]
        end
        
        imvol = abs(orientation(image_simplex))

        for i = 1:N
            for k = 1:(D + 1)
                # Get the vertices of the source simplex
                simplex[k] .= invariant_pts[triang[i][k]]
            end

            # Only compute the entry of the transfer matrix
            # if simplices are of sufficient size.
            vol = abs(orientation(simplex))

            if vol * imvol > 0 && (vol/imvol) > ϵ
                M[j, i] = intersect(simplex, image_simplex) / imvol
            end
        end
    end
    params = (; tol = tol)
    TransferOperatorApproximation(tog, M, params)
end

function transferoperator(pts, method::SimplexExact; tol::Real = 1e-8)
    tog = transopergenerator(pts, method)
    tog(tol = tol)
end


function invariantmeasure(to::TransferOperatorApproximation{G, T};
        N::Int = 200, 
        tolerance::Float64 = 1e-8, 
        delta::Float64 = 1e-8) where {G <: TransferOperator{R}, T} where R <: SimplexExact
    
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