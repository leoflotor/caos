import LinearAlgebra
import Plots
# import SparseArrays

# Use a `struct` to gather N,j,m instead of passing them individually!
# Do not oversimplify due to some variables being 1! This is just a
# very particular case.
const ω   = ω0 = 1
const γc  = 1/2 # sqrt(w * w0) / 2
const γ   = 2*γc
# const ε   = 10^-3
# const J   = 5
# const n_cursiva = 2*j

G(j) = (2 * γ) / (ω * sqrt(2*j))

# Comparing sizes
readableSize(x) = (Base.format_bytes ∘ Base.summarysize)(x)

# Factorial function for large resulting values with arbitrary precision.
myFactorial(n) = (factorial ∘ big)(n)

function cMinus(j::Int, m::Real)
  # j >= m ? sqrt(j * (j + 1) - m * (m - 1)) : 
  # ErrorException("On cMinus(): m cannot be greater than j.")
  sqrt(j * (j + 1) - m * (m - 1))
end

function cPlus(j::Int, m::Real)
  # j >= m ? sqrt(j * (j + 1) - m * (m + 1)) :
  # ErrorException("On cPlus(): m cannot be greater than j.")
  sqrt(j * (j + 1) - m * (m + 1))
end

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
                              sqrt(myFactorial(n_bra) * myFactorial(n_ket)) *
                              newG(m_bra, m_ket)^(n_bra + n_ket -2*k)
  sum_den(n_bra, n_ket, k) = myFactorial(k) * 
                              myFactorial(n_bra - k) * 
                              myFactorial(n_ket - k)
  # Cuando m_bra = m_ket la sumatoria es 0.0 a menos que n_bra = n_ket.
  # Esto se puede implementar paran evitar hacer calculos inecesarios.
  if m_bra == m_ket && n_bra != n_ket
    return 0
  else
    term1 = exp(- newG(m_bra, m_ket)^2 / 2)
    term2 = [sum_num(n_bra, n_ket, k) / sum_den(n_bra, n_ket, k) 
      for k=0:min(n_bra, n_ket)] 
    return term1 * sum(term2)
  end
end

function overlapCollection(n_max::Int, j::Int)
  n_range = 0:n_max
  m_range = -j:j
  dim = (2*j+ 1) * (n_max + 1)
  # temp = Array{NTuple{4, Float64}}(undef, dim, dim)
  temp = Array{Float64}(undef, dim, dim)
  col = 1
  for n_ket = n_range, m_ket = m_range
    row = 1
    for n_bra = n_range, m_bra = m_range
      temp[row, col] = overlap(n_bra, m_bra, n_ket, m_ket, j) # (n_bra, m_bra, n_ket, m_ket)
      row += 1
    end
    col += 1
  end
  return temp
end

# hamiltonian method
function hamiltonian(n_max::Int, j::Int)
  n_range = 0:n_max
  m_range = -j:j # Adair used it as j:-1:-j
  dim = (2*j + 1) * (n_max + 1)
  temp = zeros(dim, dim)
  # The fun begins...Iterate by column elements for better performance!
  col = 1
  for n_ket = n_range, m_ket = m_range
    row = 1
    for n_bra = n_range, m_bra = m_range
      # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
      if n_bra == n_ket && m_bra == m_ket
        temp[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
      elseif m_bra == m_ket + 1
        temp[row, col] = (- ω0 / 2) * 
          cPlus(j, m_ket) * 
          overlap(n_bra, m_bra, n_ket, m_ket, j)
      elseif m_bra == m_ket - 1
        temp[row, col] = (- ω0 / 2) * 
          cMinus(j, m_ket) * 
          overlap(n_bra, m_bra, n_ket, m_ket, j)
      end
      row += 1
    end
    col += 1
  end
  temp
end

# hamiltonian method
function hamiltonian(n_max::Int, j::Int, overlaps::AbstractArray{Float64})
  n_range = 0:n_max
  m_range = -j:j # Adair used it as j:-1:-j
  # dim = (2*j + 1) * (n_max + 1)
  dim = length(m_range) * length(n_range)
  temp = zeros(dim, dim)
  # The fun begins...Iterate by column elements for better performance!
  col = 1
  for n_ket = n_range, m_ket = m_range
    row = 1
    for n_bra = n_range, m_bra = m_range
      # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
      if n_bra == n_ket && m_bra == m_ket
        temp[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
      elseif m_bra == m_ket + 1
        temp[row, col] = (- ω0 / 2) * 
          cPlus(j, m_ket) * 
          overlaps[row, col]
      elseif m_bra == m_ket - 1
        temp[row, col] = (- ω0 / 2) * 
          cMinus(j, m_ket) * 
          overlaps[row, col]
      end
      row += 1
    end
    col += 1
  end
  temp
end

# To find converged states and their kappas
function convergenceCriterion(n_max::Int, j::Int, matrix::AbstractArray, epsilon::Float64)
  # Coefficients that correspond to the last n in every eigen vector
  nth_coefficients = n_max * (2*j + 1) + 1
  # index_of_first_nth_coeff = size(matrix)[1] - 2*j
  eigen_data = LinearAlgebra.eigen(matrix)
  # Nmax coefficients sum per state
  kappas = vec(sum(eigen_data.vectors[nth_coefficients:end,:].^2, dims=1))
  # kappas = vec(sum(eigen_data.vectors[index_of_first_nth_coeff:end,:].^2, dims=1))
  for (index, kappa) in enumerate(kappas)
    if kappa > epsilon
      return eigen_data.values[1:index], 
              eigen_data.vectors[:,1:index],
              kappas[1:index]
    end
  end
end

# for i in 1:11
#    tmp_col = tmp_vects[i:11:end,1].^2
#    findings = findall(tmp_col .> 10^-3)
#    if size(findings)[1] != 0
#        push!(tmp_vects_collect, findings[end])
#    else
#        push!(tmp_vects_collect, 0)
#    end
# end

function selectionCriterion(j::Int, matrix::AbstractArray, epsilon::Float64)
  m_range = -j:j
  # dim = (2*j + 1) * (n_max + 1)
  tmp = Array{NTuple{2, Int}}(undef, length(m_range)*size(matrix)[2])
  # tmp = []
  for (i, col) in enumerate(eachcol(matrix)), (index, m) in enumerate(m_range)
    vector = col[index:length(m_range):end].^2
    findings = findall(vector .>= epsilon)
    # If no coefficient for a given m fulfills the condition then 0 is used
    if size(findings)[1] != 0
      tmp[index + length(m_range) * (i - 1)] = (findings[end], m)
      # push!(tmp, (findings[end]-1, m))
    else
      tmp[index + length(m_range) * (i - 1)] = (0, m)
      # push!(tmp, (0, m))
    end
  end
  # tmp
  tmp_filter = Array{Tuple{Int, Int}}(undef, length(m_range))
  for (i, m) in enumerate(-j:j)
    m_filter = filter(x -> x[2] == m, tmp)
    index_max = argmax(map(x -> x[1], m_filter))
    tmp_filter[i] = m_filter[index_max]
  end
  tmp_filter
end



# scatter(F.values/5, log10.(k), xlabel=L"E_{i} / j", ylabel=L"\log_{10}(k_i)", markersize=1, primary=false)
# hline!([log10(10^-3)], label=L"10^{-3}", color=:red)

###################
# Unused
###################

function myMatrixTup(n_max, j)
  n_range = 0:n_max
  m_range = -j:j
  dim = (2*j + 1) * (n_max + 1)
  temp = Array{Tuple{Int, Int, Int, Int}}(undef, dim, dim)
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
        push!(elements, (- ω0 / 2) * cPlus(j, m_ket) * overlap(n_bra, m_bra, n_ket, m_ket, j))
      elseif m_bra == m_ket - 1
        push!(row_loc, row)
        push!(col_loc, col)
        push!(elements, (- ω0 / 2) * cMinus(j, m_ket) * overlap(n_bra, m_bra, n_ket, m_ket, j))
      end
      row += 1
    end
    col += 1
  end
  return row_loc, col_loc, elements
end

# To find max ( Nmax (mx) )
function tmpG(n_max::Int, j::Int, matrix::AbstractArray, vector_i)
  m_range = -j:j
  tmp = Array{NTuple{2, Int}}(undef,length(m_range))
  # For this particular way of iterating and problem N is the same as
  # the index of the corresponding m in the enumerate
  for (index, m) in enumerate(m_range)
    n_max_collection = argmax(matrix[index:length(m_range):end,vector_i].^2)
    tmp[index] = (n_max_collection-1, m)
  end
  tmp
end

###################
# Deprecated
###################

# function state(n_bra, m_bra, n_ket, m_ket)
#   Dict(
#     "n_bra" => n_bra,
#     "m_bra" => m_bra,
#     "n_ket" => n_ket,
#     "m_ket" => m_ket,
#     "overlap" => nothing
#   )
# end

