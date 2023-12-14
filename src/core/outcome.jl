export Outcome

"""
    Outcome <: Number
    Outcome(num::Integer)

A convenience wrapper around around an `Integer` that represents an unspecified but
enumerated outcome. It exists to distinguish the case of generic outcomes (which 
are allocated when using `counts`) vs actual integer outcomes (which may be allocated
when using `counts_and_outcomes`). It is also used for pretty-printing `Counts` and 
`Probabilities`.
"""
struct Outcome{T<:Integer} <: Number
    num::T
end
Base.show(io::IO, o::Outcome) = print(io, "Outcome($(o.num))")

# Some necessary methods for ranges to work.
Outcome{T}(x::Outcome{T}) where T<:Integer = Outcome(x.num)
Integer(x::Outcome{T}) where T<:Integer = x.num

import Base: -, +, *, rem, div, inv
for f in [:(*), :(+), :(-), :rem, :div]
    @eval begin
        @eval Base.$(f)(o1::Outcome, o2::Outcome) = Outcome($(f)(o1.num, o2.num))
        @eval Base.$(f)(o1::Outcome, o2::T) where T <: Integer = Outcome($(f)(o1.num, o2))
        @eval Base.$(f)(o1::T, o2::Outcome) where T <: Integer = Outcome($(f)(o1, o2.num))
    end
end

import Base: <, >, <=, >=, ==, isless
for f in [:(<), :(>), :(<=), :(>=), :(==), :isless]
    @eval begin
        @eval Base.$(f)(o1::Outcome, o2::Outcome) = $(f)(o1.num, o2.num)
    end
end
-(o::Outcome) = Outcome(-o.num)
inv(o::Outcome) = Outcome(inv(o.num))
Base.promote_rule(::Type{Outcome}, ::Type{Int}) = Outcome
Base.convert(::Type{Outcome}, i::Int) = Outcome(i)