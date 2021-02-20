####################################
# Abstract type
####################################

abstract type AbstractDelaunayTriangulation end

ADT = AbstractDelaunayTriangulation
Base.size(DT::ADT) = (length(DT.indices[1]) - 1, length(DT.indices))
Base.length(DT::ADT) = length(DT.indices)
dimension(DT::ADT) = length(DT.indices[1]) - 1
nsimplices(DT::ADT) = length(DT.indices)

# Indexing
Base.getindex(DT::ADT, i) = DT.indices[i]
Base.getindex(DT::ADT, i::Colon, j::Colon) = hcat(DT.indices...,)
Base.getindex(DT::ADT, i::Colon, j) = hcat(DT[j]...,)
Base.getindex(DT::ADT, i::Colon, j::Int) = hcat(DT[j])

Base.firstindex(DT::ADT) = 1
Base.lastindex(DT::ADT) = length(DT)

Base.eachindex(s::ADT) = Base.OneTo(length(s))
Base.iterate(s::ADT, state = 1) = iterate(s.indices, state)

function summarise(DT::AbstractDelaunayTriangulation)
    _type = typeof(DT)
    n_simplices = nsimplices(DT)
    D = dimension(DT)
    summary = "$D-dimensional $_type with $n_simplices simplices"
end

Base.show(io::IO, DT::AbstractDelaunayTriangulation) = println(io, summarise(DT))


export
dimension,
nsimplices
