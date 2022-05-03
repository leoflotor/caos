# Author: lft
# NOTE: Currently this deals with bigfloats due to the factorials. When the
#       data is save it is done so as bigfloats but when read it is read as 
#       float64.

# ----------------------------------------------------------------------------
# General Imports

import LinearAlgebra
import SparseArrays
import DelimitedFiles


# ----------------------------------------------------------------------------
# Constants

const ω     = ω0 = 1
const γc    = 1/2 # sqrt(w * w0) / 2
const γ     = 2*γc


# ----------------------------------------------------------------------------
# Generic functions

G(j) = (2 * γ) / (ω * sqrt(2 * j)) # what was the meaning of G?

# To compare sizes of matrices
readableSize(x) = (Base.format_bytes ∘ Base.summarysize)(x)

# Factorial function for large resulting values with arbitrary precision.
bigFactorial(n) = (factorial ∘ big)(n)

newFactorial(n) = n != 0 ? 
                    mapreduce(log, +, 1:n) |> exp : # exp(sum(log(i) for i in 1:n))
                    1

logFactorial(n) = n != 0 ?
                    sum(log(i) for i in 1:n) :
                    1

cMinus(j::Int, m::Int) = j >= abs(m) ?
    sqrt(j * (j + 1) - m * (m - 1)) :
    throw(ErrorException("The absolute value of m cannot bet greater than j."))

cPlus(j::Int, m::Int) = j >= abs(m) ? 
    sqrt(j * (j + 1) - m * (m + 1)) :
    throw(ErrorException("The absolute value of m cannot bet greater than j."))


# ----------------------------------------------------------------------------
# Overlaps in the ECB

function overlap(
    n_bra::T,
    m_bra::T,
    n_ket::T,
    m_ket::T,
    j::T
)::Float64 where T <: Int
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

    # ---------------------------------- *Methods*
    # i0 = (-1)^(n_ket) * newG^(n_bra + n_ket) * 
    #         exp(-(logFactorial(n_bra) + logFactorial(n_ket)) / 2)
    # i0 = (-1)^(n_ket) * newG^(n_bra + n_ket) / 
    #         sqrt(exp(logFactorial(n_bra) * logFactorial(n_ket)))
    # i0 = (-1)^(n_ket) * newG^(n_bra + n_ket) / 
    #         sqrt(newFactorial(n_bra) * newFactorial(n_ket))
    i0 = (-1)^(n_ket) * newG^(n_bra + n_ket) / 
            sqrt(bigFactorial(n_bra) * bigFactorial(n_ket))
    
    coeff(k) = (-1) * newG^(-2) * (n_bra - k) * (n_ket - k) / (k + 1)

    # ---------------------------------- *works*
    # (*) Helps reduce underflow... I think.
    a = i0 * coeff(0) # (*)
    summation = a
    for k in 1:min(n_bra, n_ket)-1
        a = a * coeff(k)
        summation += a
    end
    
    return exp(- newG^2 / 2) * (i0 + summation) # (*)
    # return exp(- newG^2 / 2) * i0 * (1 + summation)
end

function overlapNaive(
    n_bra::T,
    m_bra::T,
    n_ket::T,
    m_ket::T,
    j::T
) where T <: Int
    # This will deal with the special case first!
    if m_bra == m_ket && n_bra != n_ket
        return 0
    end

    newG = G(j) * (m_bra - m_ket)

    sum_num(k) = (-1)^(n_ket - k) * 
        sqrt(bigFactorial(n_bra) * bigFactorial(n_ket)) * 
        newG^(n_bra + n_ket - 2*k)
    sum_den(k) = bigFactorial(k) * 
        bigFactorial(n_bra - k) * 
        bigFactorial(n_ket - k)
    summation(k) = sum_num(k) / sum_den(k)

    term1 = exp(- newG^2 / 2)
    # term2 = sum(sum_num(k) / sum_den(k) for k=0:min(n_bra, n_ket))
    term2 = mapreduce(summation, +, 0:min(n_bra, n_ket))
    return term1 * term2
end

function overlapNaiveV2(
    n_bra::T,
    m_bra::T,
    n_ket::T,
    m_ket::T,
    j::T
) where T <: Int
    # This will deal with the special case first!
    if m_bra == m_ket && n_bra != n_ket
        return 0
    end

    newG = G(j) * (m_bra - m_ket)

    summation(k) = (-1)^(n_ket - k) * newG^(n_bra + n_ket - 2*k) * 
        sqrt(bigFactorial(n_bra) * bigFactorial(n_ket)) /
        (bigFactorial(k) * bigFactorial(n_bra - k) * bigFactorial(n_ket - k))
        
    return exp(- newG^2 / 2) * sum(summation(k) for k=0:min(n_bra, n_ket))
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
function overlapCollection(n_max::Int, j::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim = (2*j+ 1) * (n_max + 1)
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


"""
Matrix representation of the Hamiltonian for the specified `n_max` and `j`.
"""
function hamiltonian(n_max::Int, j::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim = (2*j + 1) * (n_max + 1)
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
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cPlus(j, m_ket) * 
                        overlap(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
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

function hamiltonianNaive(n_max::Int, j::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim = (2*j + 1) * (n_max + 1)
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
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cPlus(j, m_ket) * 
                        overlapNaive(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cMinus(j, m_ket) * 
                        overlapNaive(n_bra, m_bra, n_ket, m_ket, j)
                end
            end
            row += 1
        end
        col += 1
    end
    temp
end

function hamiltonianNaiveV2(n_max::Int, j::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim = (2*j + 1) * (n_max + 1)
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
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cPlus(j, m_ket) * 
                        overlapNaiveV2(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cMinus(j, m_ket) * 
                        overlapNaiveV2(n_bra, m_bra, n_ket, m_ket, j)
                end
            end
            row += 1
        end
        col += 1
    end
    temp
end

# Where k is the value of max( N( mx=0 ) )
# and x is the value of mx for a given state
# The resulting value of the parabola is the ceiling for N
p(n_max, j, k)              = j^2 / (n_max - k)
parabola(n_max, j, k, x)    = x^2 / p(n_max, j, k) + k

undesired_indices(vector_generator) = (
    index for (index, vector) in enumerate(vector_generator)
    if all(x -> x == vector[1], vector))

function hamiltonianSpecialCase(n_max::Int, j::Int, mx0::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim     = (2*j + 1) * (n_max + 1)
    temp    = zeros(dim, dim)
    conditional(n, m) = n >= parabola(n_max, j, mx0, m)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if  row >= col && !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                if n_bra == n_ket && m_bra == m_ket
                    temp[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cPlus(j, m_ket) * 
                        overlap(n_bra, m_bra, n_ket, m_ket, j)
                elseif m_bra == m_ket - 1
                    temp[row, col] = temp[col, row] = (- ω0 / 2) * 
                        cMinus(j, m_ket) * 
                        overlap(n_bra, m_bra, n_ket, m_ket, j)
                end
            end
            row += 1
        end
        col += 1
    end
    undesired_rows = undesired_indices(eachrow(temp)) |> collect
    undesired_cols = undesired_indices(eachcol(temp)) |> collect
    temp[1:end .∉ [undesired_rows], 1:end .∉ [undesired_cols]]
    # temp
end

function hamiltonianSpecialCaseOnes(n_max::Int, j::Int, mx0::Int)
    n_range = 0:n_max
    m_range = -j:j
    dim     = (2*j + 1) * (n_max + 1)
    temp    = zeros(dim, dim)
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


"""
When a matrix containing the overlaps is provided they wont be computed,
instead the overlaps in the matrix will be used.
"""
function hamiltonian(
    n_max::Int, 
    j::Int, 
    overlaps::AbstractArray
)
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
                    matrix[row, col] = matrix[col,row] = (- ω0 / 2) * 
                        cPlus(j, m_ket) * 
                        overlaps[row, col]
                elseif m_bra == m_ket - 1
                    matrix[row, col] = matrix[col, row] = (- ω0 / 2) * 
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


# Find the solution of the matrix representation of the hamiltonian
hamiltonianSolution(matrix::AbstractArray) = LinearAlgebra.eigen(matrix)


# To find converged states and their kappas
function convergenceCriterion(
    j::Int, 
    matrix::AbstractArray, 
    epsilon::Float64
)
    # Coefficients that correspond to the last n in every eigen vector
    # nth_coefficients = n_max * (2*j + 1) + 1
    index_of_first_nth_coeff = size(matrix)[1] - 2*j
    eigen_data = LinearAlgebra.eigen(matrix)
    # Nmax coefficients sum per state
    # kappas = vec(sum(eigen_data.vectors[nth_coefficients:end,:].^2, dims=1))
    kappas = vec(sum(eigen_data.vectors[index_of_first_nth_coeff:end,:].^2, dims=1))
    for (index, kappa) in enumerate(kappas)
        if kappa > epsilon
            return (values = eigen_data.values[1:index], 
                    vectors = eigen_data.vectors[:,1:index])
                    # kappas[1:index]
    end
  end
end


function convergenceCriterion(
    j::Int, 
    eigendata::NamedTuple{(:values, :vectors), Tuple{Array{T,1}, Array{T,2}}}, 
    epsilon::T
) where {T <: Float64}
    # Coefficients that correspond to the last n in every eigen vector
    index_of_first_nth_coeff = size(eigendata.values)[1] - 2*j
    kappas = vec(sum(eigendata.vectors[index_of_first_nth_coeff:end,:].^2, dims=1))
    for (index, kappa) in enumerate(kappas)
        if kappa > epsilon
            return (values = eigendata.values[1:index], 
                    vectors = eigendata.vectors[:,1:index])
                    # kappas[1:index]
    end
  end
end


function selectionCriterion(
    j::Int, 
    matrix::Array{T,2},
    epsilon::T
) where {T <: Float64}
    m_range = -j:j
    tmp = Array{NTuple{2, Int}}(undef, length(m_range)*size(matrix)[2])
    for (i, col) in enumerate(eachcol(matrix)), (index, m) in enumerate(m_range)
        vector = col[index:length(m_range):end].^2
        findings = findall(vector .>= epsilon)
        # If no coefficient for a given m fulfills the condition then 0 is used
        if size(findings)[1] != 0
            tmp[index + length(m_range) * (i - 1)] = (findings[end]-1, m)
        else
            tmp[index + length(m_range) * (i - 1)] = (0, m)
        end
    end
    # tmp
    tmp_filter = Array{Tuple{Int, Int}}(undef, length(m_range))
    for (i, m) in enumerate(-j:j)
        m_filter = filter(x -> x[2] == m, tmp)
        index_max = argmax(map(x -> x[1], m_filter))
        tmp_filter[i] = m_filter[index_max]
    end
    (ns = map(x -> x[1], tmp_filter), 
        ms = map(x -> x[2], tmp_filter))
end


# ----------------------------------------------------------------------------
# Save data

padding(var) = lpad(var, 3, "0")

file_name(file, n_max, j) = "data/$(file)_$(n_max |> padding)_$(j |> padding).csv"

function write_file(file_name::String, data::Any)
    open(file_name, "w") do io
        DelimitedFiles.writedlm(io, data, ',')
    end
end

function generate_data(
    n_max::Int, 
    j::Int, 
    eps_cvg::Float64, 
    eps_sel::Float64
)
    names = [
        "hamiltonian", 
        "eigvals", 
        "eigvects", 
        "cvgvals", 
        "cvgvects", 
        "sel"
    ]

    files = file_name.(
        names, 
        n_max, 
        j
    )
    # Check the existence of files
    files_existence = isfile.(files)
    if all(files_existence)
        return println("Data set files already exist!")
    end
    touch.(files)

    # Compute and write hamiltonian matrix
    h = hamiltonian(n_max, j)
    write_file(files[1], h)

    # Compute and write matrix eigendata
    h_sol   = hamiltonianSolution(h)
    write_file(files[2], h_sol.values)
    write_file(files[3], h_sol.vectors)
    h = nothing

    # Compute and write the matrix converged eigendata
    h_cvg   = convergenceCriterion(j, h_sol, eps_cvg)
    write_file(files[4], h_cvg.values)
    write_file(files[5], h_cvg.vectors)
    h_sol = nothing

    # Compute and write max(Nmax(mx))
    h_sel   = selectionCriterion(j, h_cvg.vectors, eps_sel)
    write_file(files[6], [h_sel.ms h_sel.ns])
    h_cvg = nothing
    h_sol = nothing

    println("Finished computing data set files !")
end


# ----------------------------------------------------------------------------
# Tips


# julia> logs = log10.(tmp_vects[1:11:end,:].^2)

# scatter(F.values/5, log10.(k), xlabel=L"E_{i} / j", ylabel=L"\log_{10}(k_i)", markersize=1, primary=false)
# hline!([log10(10^-3)], label=L"10^{-3}", color=:red)

# for i in 1:11
#    tmp_col = tmp_vects[i:11:end,1].^2
#    findings = findall(tmp_col .> 10^-3)
#    if size(findings)[1] != 0
#        push!(tmp_vects_collect, findings[end])
#    else
#        push!(tmp_vects_collect, 0)
#    end
# end


# ----------------------------------------------------------------------------
# Unused


function hamiltonianTuples(n_max::Int, j::Int)
  n_range = 0:n_max
  m_range = -j:j # Adair used it as j:-1:-j
  dim = (2*j + 1) * (n_max + 1)
  temp = Array{NTuple{4, Int}}(undef, dim, dim)
  col = 1
  for n_ket = n_range, m_ket = m_range
    row = 1
    for n_bra = n_range, m_bra = m_range
      temp[row, col] = (n_bra, m_bra, n_ket, m_ket)
      row += 1
    end
    col += 1
  end
  return temp
end

function hamiltonianSparse(n_max::Int, j::Int)
  n_range = 0:n_max
  m_range = -j:j
  # If I knoew beforehand how many elements are calculated I would be able
  # to initialize these lists instead of just appending to them.
  row_loc = Int[]
  col_loc = Int[]
  elements = Float64[]
  # The fun begins...Iterate by column elements for better performance!
  col = 1
  for n_ket = n_range, m_ket = m_range
    row = 1
    for n_bra = n_range, m_bra = m_range
      # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
      if n_bra == n_ket && m_bra == m_ket
        push!(row_loc, row)
        push!(col_loc, col)
        push!(elements, ω * (n_ket - (G(j) * m_ket)^2))
      elseif m_bra == m_ket + 1
        push!(row_loc, row)
        push!(col_loc, col)
        push!(elements, (- ω0 / 2) * 
            cPlus(j, m_ket) * 
            overlap(n_bra, m_bra, n_ket, m_ket, j))
      elseif m_bra == m_ket - 1
        push!(row_loc, row)
        push!(col_loc, col)
        push!(elements, (- ω0 / 2) * 
            cMinus(j, m_ket) * 
            overlap(n_bra, m_bra, n_ket, m_ket, j))
      end
      row += 1
    end
    col += 1
  end
  return SparseArrays.sparse(row_loc, col_loc, elements) |> SparseArrays.dropzeros
end