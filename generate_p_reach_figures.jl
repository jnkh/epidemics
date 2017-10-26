using SlurmNodes
addprocs(get_list_of_nodes())
push!(LOAD_PATH, pwd())
using PyCall
#using PyPlot,Plotting
#pygui(:qt)
using SIS,IM,PayloadGraph, Epidemics,JLD,
Dierckx,Plotting,DataAnalysis,GraphGeneration,
TwoDegree,DegreeDistribution,GraphClustering,TwoLevelGraphs
import LightGraphs
plt[:rc]("text",usetex=true)

graph_type = gamma_rg
in_parallel= true
color_range = ["b","r","g","k"]
if in_parallel
    println("running in parallel on $(nprocs()-1) nodes...")
else
    println("running in serial")
end
results = SimulationResult[]

if graph_type == regular_rg
    save_path = "../data/figure_data_regular.jld"
    N = 400
    k = 10
    alpha=0.5
    beta = 0.05
    num_trials_th = 1000
    num_trials_sim = 1000
    num_trials_sim_range = 1000*ones(Int,4) #[1000,1000,1000,100]
    k_range = [4,10,40,399]
    labels = [L"k = $k" for k in k_range]


    for (i,k) in enumerate(k_range)
        gi = get_graph_information(graph_type,N=N,k = k)
        @time si = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim_range[i],in_parallel=in_parallel)
        push!(results,si)
    end
elseif graph_type == gamma_rg
    save_path = "../data/figure_data_gamma.jld"
    N = 2000
    k = 10
    alpha= 0.666
    beta = 0.0666
    println(Epidemics.get_c_r(N,alpha,beta))
    println(Epidemics.get_n_n(N,alpha,beta)/N)

    num_trials_th = 100000
    num_trials_sim_range = 1000*ones(Int,4)
    graph_type = gamma_rg
    sigma_k_range = [1,5,15]
    labels = [latexstring("\\sigma_k = $sk") for sk in sigma_k_range]

    for (i,sigma_k) in enumerate(sigma_k_range)
        gi = get_graph_information(graph_type,N=N,k = k,sigma_k=sigma_k)
        @time si = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim_range[i],in_parallel=in_parallel)
        push!(results,si)
    end
elseif graph_type == two_level_rg
    save_path = "../data/figure_data_two_level.jld"
    N = 400
    k = 20
    alpha= 3.333
    beta = 0.3333
    println(Epidemics.get_c_r(N,alpha,beta))
    println(Epidemics.get_n_n(N,alpha,beta)/N)

    num_trials_th = 200
    num_trials_sim_range = [1000,2000,4000,8000]
    graph_type = two_level_rg
    m = k
    l_range = [19,15,10,1]
    r_range = [k-l for l in l_range]
    labels = [latexstring("k_i = $l") for l in l_range]

    for (i,l) in enumerate(l_range)
        gi = get_graph_information(graph_type,N=N,k = k,l=l,m=m,r=r_range[i])
        @time si = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim_range[i],in_parallel=in_parallel)
        push!(results,si)
    end
end

rmprocs(procs())
prepare_for_saving.(results)
save(save_path,"results",results)
d = load(save_path)
results = d["results"]

# figure(dpi=300)
# color_range = ["b","r","g","k"]
# labels = [latexstring("k = $k") for k in k_range]
# labels[4] = latexstring("k = $(k_range[4]) = N-1") 
# for (i,si) in enumerate(results)
#     plot_simulation_result(si,color=color_range[i],label=labels[i],num_points=12+i)
# end
# plot_theory_result(tr,color="k",label="well mixed",linestyle="--",num_points = 10)
# legend(fontsize=18,frameon=false,loc="lower left")
#savefig("test.png",bbox_inches="tight")
