# Author: lft
# NOTE: Currently this deals with bigfloats due to the factorials. When the
#       data is save it is done so as bigfloats but when read it is read as 
#       float64.

# ----------------------------------------------------------------------------
# General Imports

import LinearAlgebra
import SparseArrays
import DelimitedFiles
import Plots


# ----------------------------------------------------------------------------
# Constants

const ω     = ω0 = 1
const γc    = 1/2 # sqrt(w * w0) / 2
const γ     = 2*γc

# G functions that means what?
G(j) = (2 * γ) / (ω * sqrt(2*j))

# To compare sizes of matrices
readableSize(x) = (Base.format_bytes ∘ Base.summarysize)(x)

# Factorial function for large resulting values with arbitrary precision.
bigFactorial(n) = (factorial ∘ big)(n)

cMinus(j::Int, m::Real) = j >= abs(m) ?
    sqrt(j * (j + 1) - m * (m - 1)) :
    throw(ErrorException("The absolute value of m cannot bet greater than j."))

cPlus(j::Int, m::Real) = j >= abs(m) ? 
    sqrt(j * (j + 1) - m * (m + 1)) :
    throw(ErrorException("The absolute value of m cannot bet greater than j."))


# Function to compute the overlap in the BCE base.
function overlap(
    n_bra::Int,
    m_bra::Real,
    n_ket::Int,
    m_ket::Real,
    j::Int
    )
    newG(m_bra, m_ket) = G(j) * (m_bra - m_ket)
    sum_num(n_bra, n_ket, k) = (-1)^(n_ket - k) * 
        sqrt(bigFactorial(n_bra) * bigFactorial(n_ket)) * 
        newG(m_bra, m_ket)^(n_bra + n_ket -2*k)
    sum_den(n_bra, n_ket, k) = bigFactorial(k) * 
        bigFactorial(n_bra - k) * 
        bigFactorial(n_ket - k)
    if m_bra == m_ket && n_bra != n_ket
        return 0
    else
        term1 = exp(- newG(m_bra, m_ket)^2 / 2)
        # term2 = [sum_num(n_bra, n_ket, k) / sum_den(n_bra, n_ket, k) 
        #   for k=0:min(n_bra, n_ket)]
        term2 = sum(sum_num(n_bra, n_ket, k) / 
            sum_den(n_bra, n_ket, k) 
            for k=0:min(n_bra, n_ket))
        return term1 * sum(term2)
    end
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


"""
When a matrix containing the overlaps is provided they wont be computed,
instead the overlaps in the matrix will be used.
"""
function hamiltonian(n_max::Int, j::Int, overlaps::AbstractArray)
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
function hamiltonianSolution(matrix::AbstractArray)
    eigen_data = LinearAlgebra.eigen(matrix)
    (values = eigen_data.values, vectors = eigen_data.vectors)
end


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


function write_hamiltonian(n_max::Int, j::Int)
    # Check the existence of the file
    which_hamiltonian = file_name("hamiltonian", n_max, j)
    file_exists = isfile(which_hamiltonian)
    !file_exists ? touch(which_hamiltonian) : println("Hamiltonian representation was already computed!")
    
    # Compute the hamiltonian representation and save it
    data = hamiltonian(n_max, j)
    open(which_hamiltonian, "w") do io
        DelimitedFiles.writedlm(io, data, ',')
    end
end


function write_eig_data(n_max, j)
    # Assuming that (*) files already exist!
    which_hamiltonian = file_name("hamiltonian", n_max, j)      # *
    which_eig_vals = file_name("eigvals", n_max, j)
    which_eig_vects = file_name("eigvects", n_max, j)

    files = [which_eig_vals, which_eig_vects]
    files_existence = isfile.(files)
    !all(files_existence) ? touch.(files) : println("Eigendata was already computed!")

    hamiltonian = DelimitedFiles.readdlm(which_hamiltonian, ',', Float64)
    data = hamiltonianSolution(hamiltonian)

    open(which_eig_vals, "w") do io
        DelimitedFiles.writedlm(io, data.values, ',')
    end
    open(which_eig_vects, "w") do io
        DelimitedFiles.writedlm(io, data.vectors, ',')
    end
end


function write_cvg_data(n_max, j, epsilon)
    # Assuming that (*) files already exist!
    which_hamiltonian   = file_name("hamiltonian", n_max, j)    # *
    which_eig_vals      = file_name("eigvals", n_max, j)        # *
    which_eig_vects     = file_name("eigvects", n_max, j)       # *
    which_cvg_vals      = file_name("cvgvals", n_max, j)
    which_cvg_vects     = file_name("cvgvects", n_max, j)

    files = [which_cvg_vals, which_cvg_vects]
    files_existence = isfile.(files)
    !all(files_existence) ? touch.(files) : println("Converged data was already computed!")

    eig_vals = DelimitedFiles.readdlm(which_eig_vals, ',', Float64) |> vec
    eig_vects = DelimitedFiles.readdlm(which_eig_vects, ',', Float64)
    eigendata = (values = eig_vals, vectors = eig_vects)
    data = convergenceCriterion(j, eigendata, epsilon)

    open(which_cvg_vals, "w") do io
        DelimitedFiles.writedlm(io, data.values, ',')
    end
    open(which_cvg_vects, "w") do io
        DelimitedFiles.writedlm(io, data.vectors, ',')
    end
end


function write_sel_data(n_max, j, epsilon)
    # Assuming that (*) files already exist!
    which_hamiltonian   = file_name("hamiltonian", n_max, j)    # *
    which_cvg_vects     = file_name("cvgvects", n_max, j)       # *
    which_sel_data      = file_name("seldata", n_max, j)

    file_exist = isfile(which_sel_data)
    !file_exist ? touch(which_sel_data) : println("Selected data was already computed!")

    cvg_vects = DelimitedFiles.readdlm(which_cvg_vects, ',', Float64)

    data = selectionCriterion(j, cvg_vects, epsilon)
    open(which_sel_data, "w") do io
        DelimitedFiles.writedlm(io, [data.ms data.ns],  ',')
    end
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
  dim = (1*j + 1) * (n_max + 1)
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