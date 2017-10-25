addprocs(4)
push!(LOAD_PATH, pwd())
using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD,
Dierckx,Plotting,DataAnalysis,GraphGeneration,
TwoDegree,DegreeDistribution,GraphClustering,TwoLevelGraphs,
Plotting
import LightGraphs

save_path = "../data/plot.jld"

plt[:rc]("text",usetex=true)

N = 400
k = 100
alpha=0.5
beta = 0.05
num_trials_th = 1000
num_trials_sim = 1000
graph_type = regular_rg
num_trials_sim_range = [1000,1000,1000,1000]
k_range = [4,10,40]#,399]
color_range = ["b","r","g","k"]
labels = [L"k = $k" for k in k_range]
in_parallel= false

results = SimulationResult[]
if in_parallel
    println("running in parallel on $(nprocs()-1) nodes...")
else
    println("running in serial")
end
for (i,k) in enumerate(k_range)
    gi = get_graph_information(graph_type,N=N,k = k)
    @time si = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim_range[i],in_parallel=in_parallel)
    push!(results,si)
end
prepare_for_saving.(results)
save(save_path,"results",results)
d = load(save_path)
results = d["results"]

figure(dpi=200)
color_range = ["b","r","g","k"]
labels = [latexstring("k = $k") for k in k_range]
for (i,si) in enumerate(results)
    plot_simulation_result(si,color=color_range[i],label=labels[i],num_points=12+i)
end
legend(fontsize=18,frameon=false,loc="lower left")
savefig("test.png",bbox_to_anchor="full")