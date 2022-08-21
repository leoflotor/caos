"""
Dicke model module. Provides solution to Hamiltonians in the efficient coherent
basis using a differentiated approach reducing the size of the matrix by 
eliminating those column and row vectors that do not contribute to the final 
solution.

# Comments
The preliminary studies to determine the contribution of a given state where 
done by using a convergence tolerance and selection tolerance of 1E-6 and 
1E-3, respectively.
"""
module DickeModel


#==============================
Required packages
===============================#

import LinearAlgebra
import DelimitedFiles


#==============================
Parameter definitions
===============================#

const ω = 1.0
const ω0 = 1.0
const γc = sqrt(ω * ω0) / 2
const γ = 2 * γc

# The good thing about using this struct would be that the parameters
# can be changed at anytime. On the contrary, if the constants are defined
# as above, they cannot be changed unless the usr modifies the source code.
# Even if they modify the source, they would need to reload the package for
# the changes to take effect.
mutable struct Params{T<:Float64}
    ω::T
    ω0::T
    γc::T
    γ::T
    Params() = (K = new{Float64}();
        K.ω = 1.0;
        K.ω0 = 1.0;
        K.γc = sqrt(K.ω * K.ω0) / 2;
        K.γ = 2 * K.γc;
        return K)
end


#==============================
Quality of life functions
===============================#

bigFactorial(n::Int) = (factorial ∘ big)(n)

G(j::Int)::Float64 = (2 * γ) / (ω * sqrt(2 * j)) # what was the meaning of G?
# GV2(j::Int, ω::Float64, γ::Float64)::Float64 = (2 * γ) / (ω * sqrt(2 * j)) # what was the meaning of G?
# Gl(j::Int, params=Params()) = (2 * params.γ) / (params.ω * sqrt(2 * j))
# Gk(j::Int, params::Params) = (2 * params.γ) / (params.ω * sqrt(2 * j))

cMinus(j::Int, m::Int) = j >= abs(m) ?
    sqrt(j * (j + 1) - m * (m - 1)) :
    throw(ErrorException("Found j < abs(m), (j, m) = ($j, $m)."))

cPlus(j::Int, m::Int) = j >= abs(m) ?
    sqrt(j * (j + 1) - m * (m + 1)) :
    throw(ErrorException("Found j < abs(m), (j, m) = ($j, $m)."))

# Find the solution of the matrix representation of the hamiltonian
hamiltonianSolution(matrix::AbstractArray) = LinearAlgebra.eigen(matrix)

# Parabola related functions
p(n_max, j, k) = j^2 / (n_max - k)
parabola(n_max, j, k, x) = x^2 / p(n_max, j, k) + k

"""
    maxn0_approx(n_max, j, updown_factor)

Plane approximation of the dependency of `max(N(m = 0))` regarding the initial 
parameters `n_max` and `j`.
"""
maxn0_approx(n_max::Int, j::Int, updown_factor=0)::Int = ((
    0.9480705288984961 * n_max
    - 1.1205718648719307 * j
    + (-7.2622607715155185 + updown_factor))
    |> x -> round(x, RoundNearestTiesAway)
    |> Int)

# Indices where all the elements of a vector in a matrix (column or row)
# are equal to cero. These vectors do not contribute to the final solution.
undesired_indices(vector_set) = (
    index for (index, vector) in enumerate(vector_set)
    if all(x -> x == vector[1], vector))

undesired_indices_nothing(vector_set) = (
    index for (index, vector) in enumerate(vector_set)
    if all(x -> x === nothing, vector))


#==============================
Convergence, selection and 
special criteria
===============================#

function convergenceCriterion(
    j::Int,
    eigendata::LinearAlgebra.Eigen,
    epsilon::T=1E-3 # The used value on previous computations was 1E-6
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
    epsilon::T=1E-3
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

    return (ns=map(x -> x[1], tmp_filter),
        ms=map(x -> x[2], tmp_filter))
end


"""
    specialCriterion(max_indexes, eigendata, epsilon=1E-3)

This function gathers the entries in the positions given in `max_indexes` of
all the eigenvectors stored in `eigendata`, it determines which eigenvectors
have converged and returns them alongside their respective eigenvalues.

# Arguments
- `max_indexes::Array{Int, 1}`: ordered array of 1's and 0's that align with `(N(m_i), m_i)` ∀ `m_i` ∈ {-j:j} to filter those where `N(m_i)` → `maxN(m_i)`.
- `eigendata::LinearAlgebra.Eigen`: resulting eigenvalues and eigenvectors from `LinearAlgebra.eigen()`.
- `epsilon::Float64=1E-3`: default tolerance, can be changed.
"""
function specialCriterion(
    max_indexes::Array{Int,1},
    eigendata::LinearAlgebra.Eigen,
    epsilon::Float64=1E-3 # The used value on previous computations was 1E-6
)
    criterion = (
        (sum(abs2, eigendata.vectors[max_indexes.==1, :], dims=1) .< epsilon) 
        |> vec)
    return (values=eigendata.values[criterion],
        vectors=eigendata.vectors[:, criterion])
end


function coeffGrid(n_max, j, arr)
    return reshape(sum(arr.^2, dims=2), (2*j + 1, n_max + 1))
end


#==============================
Dicke model
===============================#

"""
    overlap(n_bra, m_bra, n_ket, m_ket, j)

Computes the overlap between two states given by `< N', m' | N, m >`.

# Arguments
- `n_bra::Int`: bra-state `N'`.
- `m_bra::Int`: bra-state `m'`.
- `n_ket::Int`: ket-state `N`.
- `m_ket::Int`: ket-state `m`.
- `j::Int`: pending.

# Comments
Two special cases can occur:

1) When N' = N and m' != m the `overlap()` → 0,
2) when N' = N, m' = m the `overlap()` → 1.

This function computes the overlaps in an iterative manner from an initial
term `i0` that corresponds to the first term of the summatory.

Both, the iterative method and the special cases, are implemented to reduce
memory allocations and reduce computing times.
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
    # Just renaming a factor to reduce linesize.
    newG = G(j) * (m_bra - m_ket)
    # Initial term of the summatory.
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
    overlapCollection(n_max, j)

Generates a matrix to contain all the overlaps for the intervals between 
`0:n_max` and `-j:j`.

# Arguments
- `n_max::Int`: parameter used to truncate the Hamiltonian.
- `j::Int`: pending.

# Comments
The matrix is constructed as a lower diagonal matrix and its elements are
transposed to the upper half due to the simmetry of the problem (?).

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
                (temp[row, col] 
                    = temp[col, row] 
                    = overlap(n_bra, m_bra, n_ket, m_ket, j))
            end
            row += 1
        end
        col += 1
    end
    return temp
end


function hamiltonian(
    n_max::T,
    j::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    mat = zeros(dim, dim)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if row >= col
                # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
                if n_bra == n_ket && m_bra == m_ket
                    mat[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    (mat[row, col] 
                        = mat[col, row] 
                        = (-ω0 / 2) 
                            * cPlus(j, m_ket) 
                            * overlap(n_bra, m_bra, n_ket, m_ket, j))
                elseif m_bra == m_ket - 1
                    (mat[row, col] 
                        = mat[col, row] 
                        = (-ω0 / 2) 
                            * cMinus(j, m_ket) 
                            * overlap(n_bra, m_bra, n_ket, m_ket, j))
                end
            end
            row += 1
        end
        col += 1
    end
    return mat
end


function hamiltonian(
    n_max::T,
    j::T,
    overlaps::Array{Float64,2}
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    mat = zeros(dim, dim)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            if row >= col
                # Me encuentro en el estado (n_bra, m_bra, n_ket, m_ket)
                if n_bra == n_ket && m_bra == m_ket
                    mat[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    mat[row, col] = mat[col, row] = (-ω0 / 2) *
                                                    cPlus(j, m_ket) *
                                                    overlaps[row, col]
                elseif m_bra == m_ket - 1
                    mat[row, col] = mat[col, row] = (-ω0 / 2) *
                                                    cMinus(j, m_ket) *
                                                    overlaps[row, col]
                end
            end
            row += 1
        end
        col += 1
    end
    return mat
end


#==============================
Differentiated Dicke model
===============================#

"""
    hamiltonianSpecial(n_max, j, maxN0)

Computes the Hamiltonian by employing a differentiated approach to determine
which states do contribute to the system.

# Arguments
- `n_max::Int`: parameter used to truncate the Hamiltonian.
- `j::Int`: pending. 
- `maxN0::Int`: expected value of max(Nmax(m = 0)).
"""
function hamiltonianSpecial(
    n_max::T,
    j::T,
    maxN0::T=-1,
) where {T<:Int}

    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    mat = zeros(dim, dim)  # Initialize matrix
    max_indexes = zeros(Int, dim)  # Initialize max(Nmax(m)) bool vector

    maxN0 = maxN0 == -1 ? maxn0_approx(n_max, j) : maxN0
    conditional(n, m) = n >= parabola(n_max, j, maxN0, m)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            # Only compute the bottom half of the matrix by taking advantage
            # of the symmetry of the hamiltonian
            if row >= col && !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                if n_bra == n_ket && m_bra == m_ket
                    mat[row, col] = ω * (n_ket - (G(j) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    mat[row, col] = mat[col, row] = (
                        (-ω0 / 2) *
                        cPlus(j, m_ket) *
                        overlap(n_bra, m_bra, n_ket, m_ket, j))
                elseif m_bra == m_ket - 1
                    mat[row, col] = mat[col, row] = (
                        (-ω0 / 2) *
                        cMinus(j, m_ket) *
                        overlap(n_bra, m_bra, n_ket, m_ket, j))
                end
                # The max(Nmax(m)) for each m ∈ -j:j is the same for all
                # the column vectors. Thus it is only necessary to compute it
                # once.
                if col == 1 && n_bra == (parabola(n_max, j, maxN0, m_bra) ÷ 1)
                    max_indexes[row] = 1
                end
            end
            row += 1
        end
        col += 1
    end
    # Undesired rows and columns are the same due to the simmetry of the
    # hamiltonian. But I am not certain this holds for every case.
    # NOTE. I should consider REMOVING one of them to avoid computing the same information twice.
    undesired_rows = undesired_indices(eachrow(mat)) |> collect
    undesired_cols = undesired_indices(eachcol(mat)) |> collect
    # Filtered matrix and indexes list
    mat = mat[1:end.∉[undesired_rows], 1:end.∉[undesired_cols]]
    max_indexes = max_indexes[1:end.∉[undesired_rows]] # |> x -> findall(==(1), x)
    # NOTE. How small is small to be zero?
    # mat[mat .!= 0 .&& abs.(mat) .<= 1E-16] .= 0.0
    return max_indexes, mat
end


function hamiltonianSpecialV2(
    n_max::T,
    j::T,
    maxN0::T=-1,
    params=Params()
) where {T<:Int}

    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    mat = zeros(dim, dim)  # Initialize matrix
    max_indexes = zeros(Int, dim)  # Initialize max(Nmax(m)) vector

    maxN0 = maxN0 == -1 ? maxn0_approx(n_max, j) : maxN0
    conditional(n, m) = n >= parabola(n_max, j, maxN0, m)

    col = 1
    for n_ket = n_range, m_ket = m_range
        row = 1
        for n_bra = n_range, m_bra = m_range
            # Only compute the bottom half of the matrix by taking advantage
            # of the symmetry of the hamiltonian
            if row >= col && !(conditional(n_ket, m_ket) || conditional(n_bra, m_bra))
                if n_bra == n_ket && m_bra == m_ket
                    mat[row, col] = params.ω * (n_ket - (GV2(j, params.ω, params.γ) * m_ket)^2)
                elseif m_bra == m_ket + 1
                    mat[row, col] = mat[col, row] = (-params.ω0 / 2) *
                                                    cPlus(j, m_ket) *
                                                    overlapV2(n_bra, m_bra, n_ket, m_ket, j, params.ω, params.γ)
                elseif m_bra == m_ket - 1
                    mat[row, col] = mat[col, row] = (-params.ω0 / 2) *
                                                    cMinus(j, m_ket) *
                                                    overlapV2(n_bra, m_bra, n_ket, m_ket, j, params.ω, params.γ)
                end
                # The max(Nmax(m)) for each m ∈ -j:j is the same for all
                # the column vectors. Thus it is only necessary to compute it
                # once.
                if col == 1 && n_bra == (parabola(n_max, j, maxN0, m_bra) ÷ 1)
                    max_indexes[row] = 1
                end
            end
            row += 1
        end
        col += 1
    end
    # Undesired rows and columns are the same due to the simmetry of the
    # hamiltonian. But I am not certain this holds for every case.
    # NOTE. I should consider REMOVING one of them to avoid computing the same information twice.
    undesired_rows = undesired_indices(eachrow(mat)) |> collect
    undesired_cols = undesired_indices(eachcol(mat)) |> collect
    # Filtered matrix and indexes list
    mat = mat[1:end.∉[undesired_rows], 1:end.∉[undesired_cols]]
    max_indexes = max_indexes[1:end.∉[undesired_rows]] # |> x -> findall(==(1), x)
    # NOTE. How small is small to be zero?
    # mat[mat .!= 0 .&& abs.(mat) .<= 1E-16] .= 0.0
    return max_indexes, mat
end

function hamiltonianSpecialOnes(
    n_max::T,
    j::T,
    maxN0::T
) where {T<:Int}
    n_range = 0:n_max
    m_range = -j:j
    dim = (2 * j + 1) * (n_max + 1)
    temp = zeros(dim, dim)
    conditional(n, m) = n >= parabola(n_max, j, maxN0, m)

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
    return temp
end


#==============================
Compute and save
===============================#

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

    files = file_name.(names, n_max, j)
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
    h_sol = hamiltonianSolution(h)
    write_file(files[2], h_sol.values)
    write_file(files[3], h_sol.vectors)
    h = nothing

    # Compute and write the matrix converged eigendata
    h_cvg = convergenceCriterion(j, h_sol, eps_cvg)
    write_file(files[4], h_cvg.values)
    write_file(files[5], h_cvg.vectors)
    h_sol = nothing

    # Compute and write max(Nmax(mx))
    h_sel = selectionCriterion(j, h_cvg.vectors, eps_sel)
    write_file(files[6], [h_sel.ms h_sel.ns])
    h_cvg = nothing
    h_sol = nothing

    println("Finished computing data set files !")
end

end  # module
