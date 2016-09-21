module EpidemicsSimulations

using IM,Epidemics,SIS

export save_epidemics_results

function save_epidemics_results(params)
	data_dir_path = "../data/"  #"/mnt/D/windows/MIT/classes/6/338/project/data/"

	# @everywhere begin
	N::Int = params["N"]
	alpha::Float64 = params["alpha"]
	beta::Float64 = params["beta"]
	k = params["k"]
	verbose = params["verbose"]
	fixation_threshold = params["fixation_threshold"]
	in_parallel = params["in_parallel"]
	num_trials = params["num_trials"]
	num_trials_mixed = params["num_trials_mixed"]
	graph_model = params["graph_model"]
  	graph_information = params["graph_information"]

	#function f1(x::Float64) 1 + alpha::Float64*x end
	#function f2(x::Float64)  1 + beta::Float64 end
	#function f3(x::Float64) 1 + beta + get_s_eff(x,alpha,beta,k) end
	#im_normal = InfectionModel(f1,f2);
	#im_effective = InfectionModel(f3,f2);


	im_normal = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
	im_effective = InfectionModel(x -> 1 + beta + get_s_eff(x,alpha,beta,k) , x -> 1 + beta);



	y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta,verbose)

	# end

	if myid() == 1
		println("running in parallel on $(nprocs()-1) nodes...")
		tic()
		if graph_model
		@time runs =
		run_epidemics_parallel(num_trials, () -> run_epidemic_graph(N,im_normal,graph_information,fixation_threshold),in_parallel);
		else
			if k < N-1
				@time runs =
				run_epidemics_parallel(num_trials_mixed, () -> run_epidemic_well_mixed(N,im_effective,fixation_threshold),in_parallel);
			else
				@time runs =
				run_epidemics_parallel(num_trials_mixed, () -> run_epidemic_well_mixed(N,im_normal,fixation_threshold),in_parallel);
			end
		end
		elapsed = toc()
		println("done after $elapsed seconds.")


		structure = graph_model ? "graph" : "mixed"
		filename = "epidemics_$(structure)_$(now()).jld"
		save(data_dir_path * filename,"params",params,"runs",runs)


		timing_filename = "timing_$(structure)_$(now()).jld"
		save(data_dir_path * timing_filename,"params",params,"elapsed",elapsed)

		timing_log_filename = "timing_log.out"
		f = open(timing_log_filename,"a")
		nt = graph_model ? num_trials : num_trials_mixed
		write(f,"$N $k $nt $elapsed $graph_model $(now())\n")
	end

end


end