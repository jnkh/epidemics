###Preamble###
using Distributed
import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
push!(LOAD_PATH, import_path_desai)
in_parallel = false
if in_parallel
    using SlurmNodes
	addprocs(get_list_of_nodes())
	#addprocs()
	println("Running on $(nprocs()) processes.")
else
	println("Running in serial.")
end
@everywhere using JLD2,FileIO,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,Dierckx,PyPlot,Plots,PyCall,Dates,Interpolations
# @everywhere import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
@everywhere import_path_desai = "../.././"
@everywhere push!(LOAD_PATH, import_path_desai)
# @everywhere import_path_nowak = "/n/home07/juliankh/physics/research/nowak/indirect_rec/src"
@everywhere import_path_nowak = "/Users/julian/Harvard/research/nowak/indirect_rec/src/"
@everywhere push!(LOAD_PATH, import_path_nowak)
@everywhere using Epidemics,IM, GraphGeneration,DegreeDistribution,GraphGeneration,GraphCreation,GraphClustering,TwoLevelGraphs,DataAnalysis,GraphPlotting,Plotting

use_helvetica=true
helvetica_preamble = [
raw"\renewcommand{\familydefault}{\sfdefault}",raw"\usepackage{helvet}",raw"\everymath={\sf}",
raw"\usepackage{sansmath}",   # math-font matching  helvetica
raw"\sansmath"                # actually tell tex to use it!
# r'\usepackage[helvet]{sfmath}',
#     r'\usepackage{tgheros}',    # helvetica font
#     r'\usepackage{siunitx}',    # micro symbols
#     r'\sisetup{detect-all}',    # force siunitx to use the fonts
]
rcParams = PyDict(matplotlib["rcParams"])
if use_helvetica
    rcParams["text.latex.preamble"] = helvetica_preamble
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica"]
    # from matplotlib import rcParams
else
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Times New Roman"]
end
plt.rc("text",usetex=true)
rcParams["axes.linewidth"] = 2.0
rcParams["axes.spines.right"] = false
rcParams["axes.spines.top"] = false
rcParams["lines.linewidth"] = 1.5
rcParams["font.size"] = 20

###Functions###
save_path = "../data/fig3b.jld2"
generate_data = true
plot_beta = false

if generate_data
	plot_beta = false
    sigma_k = 0
    beta = 0.04
    alpha = 0.4
    N_range = 50*2 .^collect(0:11)#collect(0:7)#[1000,2000,4000,8000,16000]
    N_min = minimum(N_range)
    N_max = maximum(N_range)
    N_dense_range = 2*Int.(round.(10.0.^range(log10(N_min),stop=log10(N_max),length=50)./2))
    k_range = [5,10,14,0]
    num_trials_sim_range = [10_000,100_000,1000_000,1000_000]
    l1 = length(k_range)
    l2 = length(N_range)
    l3 = length(N_dense_range)
    phase_trans_arr = zeros(Float64,l1,l2)
    phase_trans_simple_arr = zeros(Float64,l1,l2)
    pfix_sim_arr = zeros(Float64,l1,l2)
    pfix_th_arr = zeros(Float64,l1,l3)
    pfix_complete_th_arr = zeros(Float64,l3)
    pfix_theory_arr = zeros(Float64,l1,l2)
    pregenerate_graph = true
    # pfix_arr_small = zeros(Float64,l1,l2)
    # comm_ratio_arr = zeros(Float64,l1,l2)
    num_trials_th = 1000
    in_parallel = true
    for (i,k) in enumerate(k_range)
        num_trials_sim = num_trials_sim_range[i]
        for (j,N) in enumerate(N_range)
            if k == 0
                kloc = min(1000,N-1)
            else
                kloc = k
            end
            graph_type = sigma_k == 0 ? regular_rg : gamma_rg
            gi = get_graph_information(graph_type,N=N,k=kloc,sigma_k=sigma_k,pregenerate_graph=pregenerate_graph)
            @time sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel) 
            pfix_sim_arr[i,j] = get_pfix_sim(sr)[1]
            # pfix_th_arr[i,j] = get_pfix_th(sr)[1]

        end
        for (j,N) in enumerate(N_dense_range)
            if k == 0
                kloc = min(1000,N-1)
            else
                kloc = k
            end
            graph_type = sigma_k == 0 ? regular_rg : gamma_rg
            gi = get_graph_information(graph_type,N=N,k=kloc,sigma_k=sigma_k,pregenerate_graph=pregenerate_graph)
            tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
            pfix_th_arr[i,j] = get_pfix(tr)[1]
        end
    end

    for (j,N) in enumerate(N_dense_range)
        gi = get_graph_information(complete_rg,N=N)
        tr_complete = get_theory_result(N,alpha,beta,gi,num_trials_th)
        pfix_complete_th_arr[j] =get_pfix(tr_complete)[1]
    end
        # tr_complete.graph_information.graph=nothing
    # tr_complete.graph_information.graph_fn=nothing
	@save save_path pfix_sim_arr pfix_th_arr pfix_complete_th_arr num_trials_sim_range alpha beta N_range N_dense_range k_range
else
    @load save_path pfix_sim_arr pfix_th_arr pfix_complete_th_arr num_trials_sim_range alpha beta N_range N_dense_range k_range
end

rmprocs(workers())


###Draw Figure###
figure(figsize=(6,5))
labelsize=20
N_min = minimum(N_range);N_max = maximum(N_range) 
cm_min = 0.0
cm_max = 0.9
colors = color_range(length(k_range),"plasma",cm_min,cm_max)
for (i,k) in enumerate(k_range)
    if k == 0
        k = 1000#min(100,N-1)
    end
    num_trials_sim = num_trials_sim_range[i]
    yn = sparsity_get_yn(alpha,beta,k)#_pfix_prediction_large_N
    println(yn)
    #"-"*symbols[i]
    loglog(N_dense_range,pfix_th_arr[i,:], linestyle="-",color=colors[i],fillstyle="none",label=latexstring("k = $k"))
    errorbar(N_range,pfix_sim_arr[i,:],yerr=get_binomial_error(pfix_sim_arr[i,:],num_trials_sim),color=colors[i],marker="o",linewidth=0.5,linestyle="none",markersize=3)
#     loglog(N_range,pfix_arr[i,:], symbols[i] * "b",markersize=7,fillstyle="none",label=latexstring("k = $k"))
#     loglog(N_range,pfix_theory_arr[i,:], "-k",fillstyle="none")
#     loglog(N_range,pfix_general_arr[i,:],"-k")
    if yn < 0
#         axhline(,color="k")
        P_fix_large_N = sparsity_pfix_prediction_large_N(alpha,beta,k)
        annotate("", xy=(N_max, P_fix_large_N), xytext=(N_max*1.4, P_fix_large_N),arrowprops=Dict("facecolor"=>"k", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.04))
    end
#     if yn == 0
#         exponent = -0.5
#         loglog(N_range,pfix_arr[i,4]/N_range[4]^(exponent)*N_range.^(exponent),"--k")
#     end
    if yn > 0
#         exponent = -1.0
#         idx = findfirst(phase_trans_arr[i,:] .> 1.0)
#         if idx > 0
#             axvline(N_range[idx],color="k")
#         end
        N_critical = 2/(alpha*yn^2) #the value of N, above which there is negative selection
#         axvline(N_critical,color="k",linestyle="--",linewidth=0.5)
#         loglog(N_range,pfix_arr[i,1]/N_range[1]^(exponent)*N_range.^(exponent),"--k")
#         loglog(N_range,pfix_interm_arr[i,:],"-k")
    end
end
loglog(N_dense_range,pfix_complete_th_arr, linestyle="--",color="k",linewidth=0.8,label=latexstring("k = N-1"))
legend(frameon=false,fontsize=labelsize-3)
gca()[:tick_params](labelsize=labelsize)
# xlim([N_min,N_max])
xlabel(L"Network size, $N$",size=labelsize)
ylabel(L"Fixation probability, $P_{fix}$",size=labelsize)
ylim([1e-4,1e-1])
PyPlot.savefig("fig_3_sparsity_N_scaling.pdf",bbox_inches="tight",transparent=true)
