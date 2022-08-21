import Plots
import DelimitedFiles

colors = ["black", "orange", "green", "cyan", "red"]

padding(var) 	= lpad(var, 3, "0")
file_name(n, j) = "data/sel_$(n |> padding)_$(j |> padding).csv"
files(jval) = filter(x -> occursin("sel", x) && occursin(padding(jval) * ".csv", x), readdir("data/", join=true))

p(n_max, j, k) 				= j^2 / (n_max - k)
parabola(n_max, j, k, x) 	= x^2 / p(n_max, j, k) + k

function j_range()
	return 20:10:60
end

function write_file(file_name::String, data::Any)
    open(file_name, "w") do io
        DelimitedFiles.writedlm(io, data, ',')
    end
end

function plotMaxN()
	js = j_range()
	plot_ = Plots.plot(legendfontsize=7, legend=:bottomright)
	for (index, j) in enumerate(js[:])
        x_data = []
        y_data = []

		for i in 2:20
			n = i * j ÷ 2
			file = file_name(n, j)

			if isfile(file)
		    	data = DelimitedFiles.readdlm(file, ',')
		       	y = data[:,2]

		       	# x0 = x[(length(x)+1) ÷ 2]
		       	x0 = n
		       	y0 = y[(length(y) + 1) ÷ 2]

                append!(x_data, x0)
                append!(y_data, y0)

		    	Plots.scatter!(
		    		[x0], [y0], 
		    		# label="(j, n) = $(j), $(n)",
		    		label = i == 2 ? "j = $(j)" : "",
		       		ms = 4, 
		       		msw = 0.5,  
		       		mc = colors[index],
		       		ma = 0.35,
		       		# xticks = -50:10:50,
		       		# yticks = -10:10:160
		    		)
		    end
		end
        write_file("data/maxn_j_$(j).csv", [x_data y_data])
	end
	display(plot_)
end


function plotParabolas()
	js = j_range()
	plot_ = Plots.plot(legendfontsize=4)
	for (index, j) in enumerate(js[5])
		for i in 2:20
			n = i * j ÷ 2
			file = file_name(n, j)
			if isfile(file)
		    	data = DelimitedFiles.readdlm(file_name(n, j), ',')
		    	x = data[:,1]
		       	y = data[:,2]

		       	x0 = x[(length(x)+1) ÷ 2] |> Int
		       	y0 = y[(length(y) + 1) ÷ 2] |> Int

		       	k = y[(length(y) + 1) ÷ 2] # To get n at mx=0
		       	Plots.scatter!(
		       		x, y, 
		       		label = "(j, n) = ($(j), $(n))", 
		       		ms = 2, 
		       		msw = 0.5,  
		       		mc = colors[index],
		       		ma = 0.35,
		       		xticks = -50:10:50,
		       		yticks = -10:10:160
		    	)
		    	Plots.scatter!([x0], [y0+1], mc ="red", ms=2, label="$(y0+1)")
		       	# Adding 1 to k to include all the points below the parabola
		       	Plots.plot!(
		       		x,
		       		parabola.(n, j, k + 1, x), 
		       		# label = "$(k + 1)", 
		       		label = "",
		       		lc = colors[index],
		       		# la = 0.35
		    	)
			end
		end
	end
	display(plot_)
end
