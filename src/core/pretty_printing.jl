

"""
    relevant_fieldnames(x::T) → names::Vector{Symbol}

Internal method that returns the relevant field names to be printed for type `T`.
For example, for `Encodings`, the implementation is simply

```julia
relevant_fieldnames(e::Encoding) = fieldnames(typeof(e))
```

Individual types can override this method if special printing is desired.
"""
relevant_fieldnames(x) = fieldnames(typeof(x))

"""
    typecolor(x) → color::Symbol

The color in which to print the type name for type `typeof(x)`.

Can be used to distinguish different types, so that nested printing looks better (e.g.
`Encoding`s inside `OutcomeSpace`s).
"""
typecolor(x) = :normal

export PrintComponent
"""
    PrintComponent
    PrintComponent(s; color::Union{Symbol,Int} = :normal,
        bold::Bool = false, underline::Bool = false, blink::Bool = false,
        hidden::Bool = false, reverse::Bool = false)

Stores a string `s` and instructions for how it shall be printed. 
    
`PrintComponent`s are intended for use with `printstyled`.
"""
struct PrintComponent{S<:AbstractString}
    s::S
    color::Union{Symbol,Int}
    bold::Bool
    underline::Bool
    blink::Bool
    hidden::Bool
    reverse::Bool

    function PrintComponent(s::S; color::Union{Symbol,Int} = :normal,
        bold::Bool = false, underline::Bool = false, blink::Bool = false,
        hidden::Bool = false, reverse::Bool = false) where S
        new{S}(s, color, bold, underline, blink, hidden, reverse) 
    end
end
Base.show(io::IO, x::PrintComponent) = printstyled(io, x.s; 
    color = x.color, bold = x.bold, underline = x.underline, blink = x.blink,
    hidden = x.hidden, reverse = x.reverse)

# Stores a vector of `PrintComponent` which give instructions on how to print 
# an entire type (not just a field). The reason for having this type is so that 
# we can print nested formatted types using `PrintComponents`
struct EntireComponent
    s::Vector{PrintComponent}
end


struct PrintComponents{T <: Union{PrintComponent, EntireComponent}}
    x::Vector{T}
end


function Base.show(io::IO, x::PrintComponents) 
    for component in x.x
        show(component)
    end
end

# Modify here if more of our types should be considered.
our_types = [Encoding, OutcomeSpace]

function single_print_component!(v, name, fieldval)
    push!(v, PrintComponent("$name"; bold=true, color=:blue))
    push!(v, PrintComponent(" = "; bold=true, color=:normal))
    if any(typeof(fieldval) <: T for T in our_types)
        push!(v, printcomponents(fieldval))
    else
        push!(v, PrintComponent("$fieldval"; bold=true, color=:normal))
    end
end


# TODO: extras for certain types, e.g. {m} for OrdinalPatterns.
export printcomponents
function printcomponents(x)
    v = []
    T = typeof(x)
    N = T.name.name
    names = relevant_fieldnames(x)
    push!(v, PrintComponent("$N("))
    for (i, name) in enumerate(names)
        single_print_component!(v, name, getfield(x, name))
        if i < length(names)
            push!(v, PrintComponent(", "; bold=true, color=:normal))
        end
    end
    push!(v, PrintComponent(")"))
    return v
end

for S in Symbol.(our_types)
    @eval function Base.show(io::IO, x::$S)
        show(io, printcomponents(x))
    end
end


# for S in Symbol.(our_types)
#     @eval function print_subcomponent(io::IO, x::$S)
#         show(io, printcomponents(x))
#         # T = typeof(x)
#         # N = T.name.name
#         # names = relevant_fieldnames(x)
#         # print(io, "$N(")

#         # for (i, name) in enumerate(names)
#         #     printstyled(io, "$name"; bold=true, color=:blue)
#         #     printstyled(io, " = "; bold=true, color=:normal)
#         #     printstyled(io, "$(getfield(x, name))"; bold=true, color=:normal)
#         #     if i < length(names)
#         #         printstyled(io, ", "; bold=true, color=:normal)
#         #     end
#         # end
#         # print(io, ")")
#     end
# end

# # # 
# our_types = [Encoding, OutcomeSpace]
# for S in Symbol.(our_types)
#     @eval function Base.show(io::IO, x::$S)
#         T = typeof(x)
#         N = T.name.name
#         names = relevant_fieldnames(x)
#         fieldvals = (getfield(x, name) for name in names)

#         print(io, "$N(")
#         for (i, name) in enumerate(names)
#             printstyled(io, "$name"; bold=true, color=:blue)
#             printstyled(io, " = "; bold=true, color=:normal)
#             field = getfield(x, name)
#             #@show typeof(field)

#             # We pretty print our own types, but leave other types to default printing.
#             if typeof(field) in our_types
#                 show(printcomponents(field))
#             else
#                 printstyled(io, "$(getfield(x, name))"; bold=true, color=:normal)
#             end

#             if i < length(names)
#                 printstyled(io, ", "; bold=true, color=:normal)
#             end
#         end
#         print(io, ")")
#     end
# end