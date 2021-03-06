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
@everywhere using JLD2,FileIO,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,SimpleGraphs,Dierckx,PyPlot,Plots,PyCall,Dates,Interpolations
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
	beta = 0.1
	alpha = 1.0
	N = 10000 #10000
	k_range = [5,10,14,1000] #[5,10,12,1000]#[1000,12,10,5]
	l1 = length(k_range)
	sr_arr = Array{Any}(undef,l1)
	num_trials = 400
	num_trials_th = 100
	num_trials_sim = 100
	in_parallel = true
	pregenerate_graph = true

    trials_sim_range = [10000,100000,1000_000,1000_000]
    # num_trials_sim = 100
    in_parallel = true
    pregenerate_graph = true
    for (i,k) in enumerate(k_range)
        num_trials_sim = trials_sim_range[i]

	    gi = get_graph_information(regular_rg,N=N,k=k,carry_temporal_info=false,pregenerate_graph=pregenerate_graph)
	#     tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
	    @time    sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel)
        clean_result(sr)
	#     tr_arr[i] = tr
        # sr.graph_information.graph=nothing
        # sr.graph_information.graph_fn=nothing
	    sr_arr[i] = sr
	    println(sparsity_cascade_condition(N,alpha,beta,k))

	end
	gi = get_graph_information(complete_rg,N=N)
	tr_complete = get_theory_result(N,alpha,beta,gi,num_trials_th)
    clean_result(tr_complete)
    # tr_complete.graph_information.graph=nothing
    # tr_complete.graph_information.graph_fn=nothing
	@save save_path sr_arr tr_complete trials_sim_range alpha beta N k_range
else
	@load save_path sr_arr tr_complete trials_sim_range alpha beta N k_range
end

rmprocs(workers())


###Draw Figure###
arrow_fac = 3
labelsize=20
cm_min = 0.0
cm_max = 0.9
colors = color_range(length(k_range),"plasma",cm_min,cm_max)
# cm = matplotlib[:cm][:get_cmap]("inferno")
for (i,k) in enumerate(k_range)
    color_loc = colors[i]
    sr = sr_arr[i]
    prth = sr.prth
    intp = interpolate((prth.yy,),prth.pp,Gridded(Linear()))
    yn = sparsity_get_yn(alpha,beta,k)#_pfix_prediction_large_N
    println(yn)
    plot_simulation_result(sr,color=color_loc,label=latexstring("\$k = $k\$"),linestyle="-",num_points = 10) 

    if yn < 0
        pfix = sparsity_pfix_prediction_large_N(alpha,beta,k)
        y_crit = 1/(N*pfix)
        annotate("", xy=(y_crit,intp(y_crit)), xytext=(y_crit,intp(y_crit)*arrow_fac),arrowprops=Dict("facecolor"=>"w", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.00))
    end
    if yn == 0
        pfix = sparsity_pfix_prediction(N,alpha,beta,k)
        y_crit = 1/(N*pfix)
        annotate("", xy=(y_crit,intp(y_crit)), xytext=(y_crit,intp(y_crit)*arrow_fac),arrowprops=Dict("facecolor"=>"w", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.00))
    end
    if yn > 0
        if sparsity_cascade_condition(N,alpha,beta,k) > 1.0
            y_crit1 = yn - sqrt(yn^2 - 2/(alpha*N))
            annotate("", xy=(y_crit1, intp(y_crit1)), xytext=(y_crit1, intp(y_crit1)/arrow_fac),arrowprops=Dict("facecolor"=>"k", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.00))
        end
        y_crit2 = yn + sqrt(2/(alpha*N))
        annotate("", xy=(y_crit2, intp(y_crit2)), xytext=(y_crit2, intp(y_crit2)*arrow_fac),arrowprops=Dict("facecolor"=>"w", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.00))
    end

end
plot_theory_result(tr_complete,color="k",label=latexstring("\$k = N-1\$"),linestyle="--",num_points = 10) 
legend(frameon=false,fontsize=labelsize)
gca().tick_params(labelsize=labelsize)
# xlim([N_min,N_max])
xlabel(L"Overall frequency, y",size=labelsize)
ylabel(L"P_{reach}(y)",size=labelsize)
ylim([1e-6,1e-0])
PyPlot.savefig("fig3b.pdf",bbox_inches="tight",transparent=true)
