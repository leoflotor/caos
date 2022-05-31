"""
Dicke model module. Provides solution to matrices in the efficient coherent
basis using a differentiated approach to contruct the matrix.
"""
module DickeModel

import LinearAlgebra

const ω = 1
const ω0 = 1
const γc = sqrt(ω * ω0) / 2
const γ = 2 * γc

G(j) = (2 * γ) / (ω * sqrt(2 * j)) # what was the meaning of G?

bigFactorial(n) = (factorial ∘ big)(n)

cMinus(j::Int, m::Int) = j >= abs(m) ?
                         sqrt(j * (j + 1) - m * (m - 1)) :
                         throw(ErrorException("The absolute value of m cannot bet greater than j."))

cPlus(j::Int, m::Int) = j >= abs(m) ?
                        sqrt(j * (j + 1) - m * (m + 1)) :
                        throw(ErrorException("The absolute value of m cannot bet greater than j."))

# Find the solution of the matrix representation of the hamiltonian
hamiltonianSolution(matrix::AbstractArray) = LinearAlgebra.eigen(matrix)

# Parabola related functions
p(n_max, j, k) = j^2 / (n_max - k)
parabola(n_max, j, k, x) = x^2 / p(n_max, j, k) + k

# Indices where all the elements of a vector in a matrix (column or row)
# are equal to cero. These vectors do not contribute to the final solution.
undesired_indices(vector_set) = (
    index for (index, vector) in enumerate(vector_set)
    if all(x -> x == vector[1], vector))

undesired_indices_nothing(vector_set) = (
    index for (index, vector) in enumerate(vector_set)
    if all(x -> x == nothing, vector))


"""
Computes a single overlap for a given < N',m' | N,m >.

# Arguments
- `n_bra::Integer`: pending
- `m_bra::Integer`: pending
- `n_ket::Integer`: pending
- `m_ket::Integer`: pending
- `j::Integer`: pending

# Comments
Two special cases can occur which this function is capable to deal with:

1) N' = N, m' != m

2) N' = N, m' = m

This function computes the overlaps in an iterative manner from an initial
term `i0` that corresponds to the first term of the summatory.
"""
function overlap(
    n_bra::T,
    m_bra::T,
    n_ket::T,
    m_ket::T,
    j::T
)::Float64 where {T<:Int}
    # Dealing with the special cases first!
    # 1st special case N' = N, m' != m.
    if m_bra == m_ket && n_bra != n_ket
        return 0
    end
    # 2nd special case N' = N, m' = m.
    if m_bra == m_ket && n_bra == n_ket
        return 1
    end

    newG = G(j) * (m_bra - m_ket)
    i0 = (-1)^(n_ket) * newG^(n_bra + n_ket) /
         sqrt(bigFactorial(n_bra) * bigFactorial(n_ket))

    coeff(k) = (-1) * newG^(-2) * (n_bra - k) * (n_ket - k) / (k + 1)

    a = i0 * coeff(0)
    summation = a
    for k in 1:min(n_bra, n_ket)-1
        a = a * coeff(k)
        summation += a
    end

    return exp(-newG^2 / 2) * (i0 + summation)
end


"""
Generates a matrix to contain all the overlaps for the intervals between 
`0:n_max` and `-j:j`.

The matrix is constructed as a lower diagonal matrix and its elements are
transposed to the upper half due to the simmetry of the problem (?).

# Arguments
- `n_max::Integer`: pending
- `j::Integer`: pending

# Comments
Maybe the upper half is unnecessary.
"""
function overlapCollection(
    n_max::T,
    j::T
)::Array{Float64,2} where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = Array{Float64}(undef, dim, dim)
    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if row >= col
                temp[row, col] = temp[col, row] = overlap(n_bra, m_bra, n_ket, m_ket, j)
            end
            row += 1
        end
        col += 1
    end
    temp
end


function hamiltonian(
    n_max::T,
    j::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = zeros(dim, dim)
    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if row >= col
                # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
                if n_bra == n_ket && m_bra == m_ket
                    temp[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    temp[row, col] = temp[col, row] = (-ω0 / 2) *
                                                      cPlus(j, m_ket) *
                                                      overlap(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (-ω0 / 2) *
                                                      cMinus(j, m_ket) *
                                                      overlap(n_bra, m_bra, n_ket, m_ket, j)
                end
            end
            row += 1
        end
        col += 1
    end
    temp
end


function hamiltonian(
    n_max::T,
    j::T,
    overlaps::Array{Float64,2}
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    # dim = (2*j + 1) * (n_max + 1)
    dim = length(m_range) * length(n_range)
    matrix = zeros(dim, dim)
    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if row >= col
                # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
                if n_bra == n_ket && m_bra == m_ket
                    matrix[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    matrix[row, col] = matrix[col, row] = (-ω0 / 2) *
                                                          cPlus(j, m_ket) *
                                                          overlaps[row, col]
                elseif m_bra == m_ket - 1
                    matrix[row, col] = matrix[col, row] = (-ω0 / 2) *
                                                          cMinus(j, m_ket) *
                                                          overlaps[row, col]
                end
            end
            row += 1
        end
        col += 1
    end
    matrix
end


function convergenceCriterion(
    j::Int,
    eigendata::LinearAlgebra.Eigen{Float64,Float64,Matrix{Float64},Vector{Float64}},
    epsilon::T=10^-3
) where {T<:Float64}
    # Coefficients that correspond to the last n in every eigen vector
    index_of_first_nth_coeff = size(eigendata.values)[1] - 2 * j
    # kappas = vec(sum(eigendata.vectors[index_of_first_nth_coeff:end, :] .^ 2, dims=1))
    kappas = sum(abs2, eigendata.vectors[index_of_first_nth_coeff:end, :], dims=1) |> vec
    for (index, kappa) in enumerate(kappas)
        if kappa > epsilon
            return (values=eigendata.values[1:index],
                vectors=eigendata.vectors[:, 1:index],
                kappa=kappas[1:end])
        end
    end
end


function selectionCriterion(
    j::Int,
    matrix::Array{T,2},
    epsilon::T=10^-3
) where {T<:Float64}
    m_range = -j:j
    tmp = Array{NTuple{2,Int}}(undef, length(m_range) * size(matrix)[2])

    for (i, col) in enumerate(eachcol(matrix)), (index, m) in enumerate(m_range)
        vector = col[index:length(m_range):end] .^ 2
        findings = findall(vector .>= epsilon)
        # If no coefficient for a given m fulfills the condition then 0 is used
        if size(findings)[1] != 0
            tmp[index+length(m_range)*(i-1)] = (findings[end] - 1, m)
        else
            tmp[index+length(m_range)*(i-1)] = (0, m)
        end
    end

    # tmp
    tmp_filter = Array{Tuple{Int,Int}}(undef, length(m_range))
    for (i, m) in enumerate(-j:j)
        m_filter = filter(x -> x[2] == m, tmp)
        index_max = argmax(map(x -> x[1], m_filter))
        tmp_filter[i] = m_filter[index_max]
    end

    (ns=map(x -> x[1], tmp_filter),
        ms=map(x -> x[2], tmp_filter))
end


function hamiltonianSpecialCase(
    n_max::T,
    j::T,
    mx0::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = zeros(dim, dim) # store matrix entries
    max_indexes = zeros(Int8, dim) # store indexes that correspond to max(Nmax(m))
    conditional(n, m) = n >= parabola(n_max, j, mx0, m)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            # Only compute the bottom half of the matrix by taking advantage
            # on the symmetry of the hamiltonian
            if row >= col && !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                if n_bra == n_ket && m_bra == m_ket
                    temp[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    temp[row, col] = temp[col, row] = (-ω0 / 2) *
                                                      cPlus(j, m_ket) *
                                                      overlap(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (-ω0 / 2) *
                                                      cMinus(j, m_ket) *
                                                      overlap(n_bra, m_bra, n_ket, m_ket, j)
                end
                if col == 1 && n_bra == (parabola(n_max, j, mx0, m_bra) ÷ 1)
                    max_indexes[row] = 1
                end
            end
            row += 1
        end
        col += 1
    end
    # Undesired rows and columns are the same due to the simmetry of the
    # hamiltonian.
    # NOTE. I should consider REMOVING one of them to avoid computing the same information twice.
    undesired_rows = undesired_indices(eachrow(temp)) |> collect
    undesired_cols = undesired_indices(eachcol(temp)) |> collect
    # Filtered matrix and indexes list
    temp = temp[1:end.∉[undesired_rows], 1:end.∉[undesired_cols]]
    max_indexes = max_indexes[1:end.∉[undesired_rows]] # |> x -> findall(==(1), x)

    # temp[temp .!= 0 .&& abs.(temp) .<= 1E-16] .= 0.0
    max_indexes, temp
end

function hamiltonianSpecialCaseOnes(
    n_max::T,
    j::T,
    mx0::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = zeros(dim, dim)
    conditional(n, m) = n >= parabola(n_max, j, mx0, m)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                temp[row, col] = 1
            end
            row += 1
        end
        col += 1
    end
    # undesired_rows = undesired_indices(eachrow(temp)) |> collect
    # undesired_cols = undesired_indices(eachcol(temp)) |> collect
    # temp[1:end .∉ [undesired_rows], 1:end .∉ [undesired_cols]]
    temp
end


mutable struct Braket
    n_bra::Int
    m_bra::Int
    m_ket::Int
    n_ket::Int
    entry::Float64
    falsy::Bool     # Truth value depending if this entry corresponds to a value where max( Nmax( m ) ) is found
end


function hamiltonianBrakets(
    n_max::T,
    j::T,
    mx0::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = Array{Union{Braket,Nothing}}(nothing, dim, dim)
    conditional(n, m) = n >= parabola(n_max, j, mx0, m) # To filter states

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            # If the condition is satisfied then the element is filled with
            # its state information
            state = Braket(n_bra, m_bra, n_ket, m_ket, 0.0, false)
            if !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                temp[row, col] = state
                # More conditions
                if n_bra == n_ket && m_bra == m_ket
                    temp[row, col].entry = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    temp[row, col].entry = (-ω0 / 2) *
                                           cPlus(j, m_ket) *
                                           overlap(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col].entry = (-ω0 / 2) *
                                           cMinus(j, m_ket) *
                                           overlap(n_bra, m_bra, n_ket, m_ket, j)
                end
                # Only for the first column because for all m's this process
                # will render the same max(Nmax(m)) respecively
                if col == 1 && n_bra == (parabola(n_max, j, mx0, m_bra) ÷ 1)
                    temp[row, col].falsy = true
                end
            end
            row += 1
        end
        col += 1
    end
    undesired_rows = undesired_indices_nothing(eachrow(temp)) |> collect
    temp[1:end.∉[undesired_rows], 1:end.∉[undesired_rows]]
end

end # module
