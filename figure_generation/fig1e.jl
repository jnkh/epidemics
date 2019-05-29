###Preamble###
using Distributed
push!(LOAD_PATH, pwd()*"/..")
using SlurmNodes
in_parallel = true
if in_parallel
	# addprocs(get_list_of_nodes())
	addprocs()
	println("Running on $(nprocs()) processes.")
else
	println("Running in serial.")
end
@everywhere using JLD,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,Dierckx,PyPlot,Plots,PyCall,Dates
@everywhere push!(LOAD_PATH, pwd()*"/..")
@everywhere import_path = "/Users/julian/Harvard/research/nowak/indirect_rec/src"
@everywhere push!(LOAD_PATH, import_path)
@everywhere using Epidemics,IM, GraphGeneration,DegreeDistribution,GraphGeneration,GraphCreation,GraphClustering,TwoLevelGraphs,DataAnalysis,GraphPlotting

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
else
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Times New Roman"]
end
plt.rc("text",usetex=true)

###Functions###
function get_graphs_clustering_range(C_begin,C_end,num_graphs)
	C_range = range(C_begin,stop=C_end,length=num_graphs)
	C_range_actual = zeros(length(C_range))
	g_range = Array{Any}(undef,length(C_range))
	overlap_arr = zeros(length(C_range))
	for (i,C_curr) in enumerate(C_range)
	    g1 = LightGraphs.copy(g)
	    change_clustering_by_swapping(g1,C_curr)
	    @assert(length(connected_components(g1)) == 1)
	    overlap_arr[i] = community_overlap(g,g1,400)[1]
	    g_range[i] = g1
	    C_range_actual[i] = mean(local_clustering_coefficient(g1))
	    println("C: $C_curr, C_actual: $(C_range_actual[i])")
	end
	return g_range,C_range_actual,overlap_arr
end


###Generate Data###
num_trials = 40_000
N = 4100
k = 50
C = 0.5
sigma_k = 8
min_degree = 3
g_fb = create_graph(N,k,:fb,C)
if N >= 4000
	g = g_fb
else
	g = subsample_from_communities(g_fb,2*Int(round(N/2)),k)
end
C = mean(local_clustering_coefficient(g))
k = mean(degree(g))
N = nv(g)
C_random = k/N
println("N: $N, k: $k, C: $C")
@assert(length(connected_components(g)) == 1)

save_path_graphs = "./data/fig1e_graphs.jld"#"./data/fig2e_$(now()).jld"
if isfile(save_path_graphs)
	dat = load(save_path_graphs)
	g_range = dat["g_range"]
	C_range_actual = dat["C_range_actual"]
	overlap_arr = dat["overlap_arr"]
else
	g_range,C_range_actual,overlap_arr = get_graphs_clustering_range(1.5*C_random,C,10)
	save(save_path_graphs,"g_range",g_range,"C_range_actual",C_range_actual,"overlap_arr",overlap_arr)
end

generate_data = true
save_path = "./data/fig1e.jld"#"./data/fig2e_$(now()).jld"

if generate_data
	use_theory = false
	num_trials_th = 100
	trials_range = num_trials*ones(Int,length(C_range_actual))#4000*ones(Int,length(C_range))#[40000,40000,40000,4000,2000]#10000*ones(Int,5)#[1000,1000,1000,1000,1000]
	ab_range = [(1.0,0.2),(0.0,-0.02),(0.0,0.0),(0.0,0.003)]
	pfix_range = zeros(length(C_range_actual),length(ab_range))
	yy_arr = Array{Any}(undef,length(C_range_actual),length(ab_range))
	pp_arr = Array{Any}(undef,length(C_range_actual),length(ab_range))
	trials_arr = zeros(length(C_range_actual),length(ab_range))

	for (i,C_curr) in enumerate(C_range_actual)
	    for (j,(alpha,beta)) in enumerate(ab_range)
	    	if beta < 0.0
	    		num_trials = 40_000
	    	elseif alpha > 0.0 && C_curr > 0.45
	    		num_trials = 40_000
	    	else
	    		num_trials = trials_range[i]
	    	end
	        # end
	        g1 = g_range[i]
	        gi = get_graph_information(custom_rg,custom_g=g1)
	        @time yy,pp,pfix,_ = Epidemics.get_simulation_yy_pp(gi,nv(g1),alpha,beta,num_trials_th,num_trials_sim=num_trials,in_parallel=in_parallel,use_theory=use_theory)
	        pfix_range[i,j] = pfix
	        yy_arr[i,j] = yy
	        pp_arr[i,j] = pp
	        trials_arr[i,j] = num_trials
	    end
	end

	save(save_path,"yy_arr",yy_arr,"pp_arr",pp_arr,"trials_arr",trials_arr,
	"pfix_range",pfix_range,"ab_range",ab_range)
else
	dat = load(save_path)
	yy_arr = dat["yy_arr"]
	pp_arr = dat["pp_arr"]
	trials_arr = dat["trials_arr"]
	pfix_range = dat["pfix_range"]
	ab_range = dat["ab_range"]
end

rmprocs(workers())


###Draw Figure###
figure(figsize=(6,6))
fontsize=20
cm_max = 0.85
cm_min = 0.25
cm = matplotlib[:cm][:get_cmap]("inferno")

# font2 = Dict(["color"=>"blue","fontsize"=>fontsize])
# ylabel("Community overlap",fontdict=font2)
# setp(gca()[:get_yticklabels](),color="blue")
# gca()[:tick_params](labelsize=fontsize)
# PyPlot.plot(C_range_actual,overlap_arr,"-b")
# ylim([0,1])
# ax2 = gca()[:twinx]()
xlabel(L"Clustering, $C$",fontsize=fontsize)
gca()[:set_xticks](0.0:0.1:0.6)
ylabel(L"Fixation probability, $P_{fix}$",fontsize=fontsize)
gca()[:tick_params](labelsize=fontsize)
for (j,(alpha,beta)) = enumerate(ab_range)
    color = cm(cm_min + (cm_max-cm_min)*(j-1)/length(ab_range))
    tmp = pfix_range[:,j]
    fixed_idx = tmp .> 0.0
    tmp[.~fixed_idx] = 1.0./trials_arr[.~fixed_idx,j]# 1.0/40_000
    contagion_word = alpha != 0 ? "Complex" : "Simple" 
    PyPlot.semilogy(C_range_actual[fixed_idx],tmp[fixed_idx],"-o",color=color,label=latexstring("$(contagion_word), \$\\alpha=$alpha,\\beta=$beta\$"))#\beta = $beta, \alpha=$alpha\$"))
    PyPlot.semilogy(C_range_actual[.~fixed_idx],tmp[.~fixed_idx],"-o",color=color,fillstyle="none")
    tmp_err = get_binomial_error(tmp,trials_arr[:,j])
    println(tmp_err)
    PyPlot.errorbar(C_range_actual[fixed_idx],tmp[fixed_idx],linestyle="none",yerr=0.5*tmp_err[fixed_idx],color=color)
end
ylim([2e-5,2.5e-2])
legend(fontsize=fontsize-5,frameon=false,loc=(0.37,0.2))
gca().spines["right"].set_visible(false)
gca().spines["top"].set_visible(false)
PyPlot.savefig("fig2e.pdf",bbox_inches="tight",transparent=true)
