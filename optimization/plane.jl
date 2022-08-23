import DelimitedFiles

padding(jval) = lpad(jval, 3, "0")

files(jval) = filter(x -> occursin("sel", x) && occursin(padding(jval) * ".csv", x), readdir("data/", join=true))

function data()
    jrange = 10:10:60
    data = Dict()
    for j in jrange
        maxN0s, Nmaxs = Int64[], Int64[]
        for file in files(j)
            tmp = DelimitedFiles.readdlm(file, ',')
            maxN0 = (indexin(0, tmp[:,1])
                     |> x -> tmp[:,2][x]
                     |> x -> Int(x[1]))
            Nmax = (file
                    |> x -> findall('_', x)
                    |> x -> (x[1]+1 : x[2]-1)
                    |> x -> file[x]
                    |> x -> parse(Int, x))
            push!(maxN0s, maxN0)
            push!(Nmaxs, Nmax)
        end
        data["j" * padding(j)] = Dict("j" => j,
                                      "maxN0" => maxN0s,
                                      "Nmax" => Nmaxs)
    end
    return data
end
