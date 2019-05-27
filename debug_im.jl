push!(LOAD_PATH, pwd()*"/.")
import_path = "/Users/julian/Harvard/research/nowak/indirect_rec/src"
push!(LOAD_PATH, import_path)
using LightGraphs, Epidemics, GraphGeneration,IM
using GraphGeneration
using NLsolve
using Distributions
using DegreeDistribution
using StatsBase
using GraphCreation
using Clustering
using DataAnalysis
using TwoLevelGraphs
using PyCall, PyPlot
use_helvetica = true

helvetica_preamble = [
raw"\renewcommand{\familydefault}{\sfdefault}",raw"\usepackage{helvet}",raw"\everymath={\sf}",
raw"\usepackage{sansmath}",   # math-font matching  helvetica
raw"\sansmath"                # actually tell tex to use it!
# r'\usepackage[helvet]{sfmath}',
#     r'\usepackage{tgheros}',    # helvetica font
#     r'\usepackage{siunitx}',    # micro symbols
#     r'\sisetup{detect-all}',    # force siunitx to use the fonts
]
if use_helvetica
    PyCall.PyDict(matplotlib["rcParams"])["text.latex.preamble"] = helvetica_preamble
    PyCall.PyDict(matplotlib["rcParams"])["font.family"] = "sans-serif"
    PyCall.PyDict(matplotlib["rcParams"])["font.sans-serif"] = ["Helvetica"]
    # from matplotlib import rcParams
else
    PyCall.PyDict(matplotlib["rcParams"])["font.family"] = "serif"
    PyCall.PyDict(matplotlib["rcParams"])["font.sans-serif"] = ["Times New Roman"]
end
plt[:rc]("text",usetex=true)


using LightGraphs, IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, PyPlot, Distributions
using CliquePercolation
plt[:rc]("text",usetex=true)

function get_alpha_beta(N,c_r,y_n)
    n_n = Int(N*y_n)#y_n*N
    beta = get_beta(N,c_r,n_n)#4.0/(c_r*n_n)
    alpha = get_alpha(N,c_r,n_n)#(N*beta)/n_n
    return alpha,beta
end
    

function get_t(N,k,m,l)
    r = k -l
    t = TwoLevel(Int(ceil(N/m)*m),m,l,r)
    return t
end

# function get_community_graph_fixation_ratio(t,alpha,beta)
#     return get_fixation_MC(t,alpha,beta)/get_fixation_MC(t,alpha,beta,true)
# #     yy,pp,pp_reverse = get_single_community_fixation(t,alpha,beta,false)
# #     return pp[end]/pp_reverse[end]
# end

function get_theory_sim_fixation_ratio(t,alpha,beta,num_trials,use_model = false)
    success = true
    if use_model
        yys,pps,_ = TwoLevelGraphs.get_p_reach_theory(t,alpha,beta,t.N,true,200)
    else
        yys,pps = get_simulation_yy_pp(t,alpha,beta,num_trials)
    end
#     yy,pp,pp_reverse = get_single_community_fixation(t,alpha,beta)
#     pfixth = pp[end]
    pfixth = get_fixation_MC(t,alpha,beta)
    pfix = pps[end]
    if yys[end] < 0.99
        println("simulation didn't reach 1.0")
        println("theory suggests at least $(1/pfixth)")
        success = false
        pfix = 0
#         return -1
    end
    figure(1)
    loglog(yys,pps)
    return pfix/pfixth, success,pfix,pfixth
end


using Dierckx

function get_simulation_yy_pp(t,alpha,beta,num_trials=100)
    im_normal = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
    fixation_threshold = 1.0

    verbose = false
    ###Set to true if we want by-node information on infecteds (much more data!)
    carry_by_node_information = false
    graph_model = true
    in_parallel = true
    T = generate_regular_two_level_graph(t)
    graph_fn = () -> T 
    graph_data = TwoLevelGraph(LightGraphs.Graph(),t,get_clusters(t))
    graph_information = GraphInformation(graph_fn,Graph(),carry_by_node_information,graph_data,two_level_rg)

    # @time runs = run_epidemics_parallel(num_trials,() -> run_epidemic_graph(N,im_normal,graph_information,fixation_threshold),in_parallel);
    # yy,pp = get_p_reach(runs)
    # yy /= N
    @time runs = run_epidemics_parallel(num_trials,() -> run_epidemic_graph_gillespie(t.N,im_normal,graph_information,fixation_threshold),in_parallel);
    yys,pps = get_p_reach(runs)
    yys /= N;
    return yys,pps
end



function get_dmp_dmm(t,j,alpha,beta)
    i_orig = t.i
    t.i = i_orig + j 
    z_s = TwoLevelGraphs.get_y_local(t,j,true)
    z_sq_s = TwoLevelGraphs.get_y_squared_local(t,j,true)
    z_i = TwoLevelGraphs.get_y_local(t,j,false)
#     println("j: $j")
#     println("z_s: $(z_s)")
#     println("z_i: $(z_i)")
#     println()

    y_i = j/m
    
    
    dmp = (1-y_i)*(z_s + alpha*z_sq_s)
    dmm = (y_i)*(1-z_i)*(1 + beta)
    
    if j == 0 || j == t.m
        dmm = 0 #j == 0
        dmp = 0 #j ==t.m
    end

    t.i = i_orig
    return dmp,dmm
end

function get_fixation_MC(t,alpha,beta,reverse=false)
    y = 0.0
    t.i = t.N*y

    j_range = collect(0:t.m)
    y_range = j_range/t.m
    s_arr = zeros(Float64,size(j_range))
    dmp_arr = zeros(Float64,size(j_range))
    dmm_arr = zeros(Float64,size(j_range))
    s_plus_arr = zeros(Float64,size(j_range))
    for (idx,j) in enumerate(j_range)
        dmp,dmm = get_dmp_dmm(t,j,alpha,beta)
        dmp_arr[idx] = dmp
        dmm_arr[idx] = dmm
    end
    if reverse
        tmp = dmp_arr
        dmp_arr = dmm_arr
        dmm_arr = tmp
        dmp_arr = dmp_arr[end:-1:1]
        dmm_arr = dmm_arr[end:-1:1]
    end
        
    M = zeros(Float64,m+1,m+1)
    Q = zeros(Float64,m-1,m-1)
#     for (idx,j) in enumerate(j_range)
    for idx in 2:m
        M[idx,idx+1] = dmp_arr[idx]
        M[idx,idx-1] = dmm_arr[idx]
        M[idx,idx] = -dmp_arr[idx]-dmm_arr[idx]
    end
    Q = M[2:m,2:m]
    W1 = M[2:m,1]
    W2 = M[2:m,m+1]
    return(- inv(Q)*W2)[1]
    
#     sanity check
#     eps = 0.5
#     P = eye(m+1) + eps*M
#     s0 = zeros(m+1)
#     s0[2] = 1.0
#     println((s0'*(P^100000))[end])
    
#     tot = 0
#     prod_vec = zeros(m)
#     lambdas = dmp_arr[2:m]
#     mus = dmm_arr[2:m]
#     for i in 1:m
#         prod_vec[i] = prod(lambdas[1:i-1])*prod(mus[i:end])
#     end
# #     prod_vec[1] = prod(mus)
# #     prod_vec[end] = prod(lambdas)
#     return prod(lambdas)/sum(prod_vec)
        
#     M[1,1] = 1
#     M[end,end] = 1
#     pi = eig(M')
    
#     return pi
end

function get_single_community_fixation(t,alpha,beta,plotting=false)
    y = 0.0
    t.i = t.N*y

    j_range = collect(0:t.m)
    y_range = j_range/t.m
    s_arr = zeros(Float64,size(j_range))
    dmp_arr = zeros(Float64,size(j_range))
    dmm_arr = zeros(Float64,size(j_range))
    s_plus_arr = zeros(Float64,size(j_range))
    for (idx,j) in enumerate(j_range)
        dmp,dmm = get_dmp_dmm(t,j,alpha,beta)
        s_arr[idx] = (dmp-dmm)
        s_plus_arr[idx] = (dmp+dmm)
        dmp_arr[idx] = dmp
        dmm_arr[idx] = dmm
    end

    # plot(j_range,s_arr./s_plus_arr)
    # plot(j_range,s_arr,"-")
    if plotting
        plot(j_range,dmm_arr,"--")
        plot(j_range,dmp_arr,"-")
    end


    interpolation_order = 3
    s_fn(x) = evaluate(Spline1D(y_range,s_arr,k=interpolation_order,bc="extrapolate"),x)
    splus_fn(x) = evaluate(Spline1D(y_range,s_plus_arr,k=interpolation_order,bc="extrapolate"),x)

    s_m_fn(x) = evaluate(Spline1D(1-y_range[end:-1:1],-s_arr[end:-1:1],k=interpolation_order,bc="extrapolate"),x)

    
    if plotting
        figure()
        plot(y_range,s_fn(y_range))
        plot(y_range,s_m_fn(y_range))
    end
    yy = y_range[1:end]
    pp_reverse = P_reach_fast(s_m_fn,splus_fn,t.m,1.0/t.m,yy)
    pp = P_reach_fast(s_fn,splus_fn,t.m,1.0/t.m,yy)
    return yy,pp,pp_reverse
end

## dependence on two variables
plot_beta = true
N = 10000;k=20;m=20;l=19;r = k-l
N_small = 400
# N_range = [120,200,320,440]#,800,1600,3200]
alpha_range = [0.025,0.05]#5*[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.2,0.3]#collect(0.01:0.1:0.5)
if plot_beta
    l = 19
    beta_range = [0.05,0.1]#collect(0.05:0.05:0.2)#[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.2,0.3]#collect(0.01:0.1:0.5)
    l2 = length(beta_range)
else
    beta = 0.01
    l_range = [4,6,8,10,12,14,16,17,18,19] 
    l2 = length(l_range)
end
l1 = length(alpha_range)
# l2 = length(N_range)
phase_trans_arr = zeros(Float64,l1,l2)
pfixth_arr = zeros(Float64,l1,l2)
pfix_arr = zeros(Float64,l1,l2)
pfix_arr_small = zeros(Float64,l1,l2)
comm_ratio_arr = zeros(Float64,l1,l2)
succ_arr = zeros(Bool,l1,l2)
# beta = 0.05
num_trials = 400
num_trials_th = 40
for (i,alpha) in enumerate(alpha_range)
#     for (j,l) in enumerate(l_range)
    for (j,beta) in enumerate(beta_range)
        r = k-l
#     for (j,N) in enumerate(N_range)
        t = get_t(N,k,m,l)
        gi = get_graph_information(two_level_rg,N=N,k=k,m=m,l=l,r=r)
        @time _,_,pfix,_ = Epidemics.get_simulation_yy_pp(gi,N,alpha,beta,num_trials_th,num_trials_sim=num_trials,use_theory=true)
        # pfix_arr[i,j] = pfix
        # phase_trans_arr[i,j] = phase_transition_condition(t,alpha,beta)
        # comm_ratio_arr[i,j] = get_community_graph_fixation_ratio(t,alpha,beta)
        # N_small = Int(round(N/10))
        # t = get_t(N_small,k,m,l)
        # gi = get_graph_information(two_level_rg,N=N_small,k=k,m=m,l=l,r=r)
        # _,_,pfix,_ = Epidemics.get_simulation_yy_pp(gi,N_small,alpha,beta,num_trials_th,num_trials_sim=num_trials,use_theory=true)
        # pfix_arr_small[i,j] = pfix
#         println(comm_ratio_arr[i,j])
#         sim_ratio, succ,pfix,pfixth = get_theory_sim_fixation_ratio(t,alpha,beta,num_trials)
#         sim_ratio_arr[i,j] = sim_ratio
#         succ_arr[i,j] = succ
#         pfixth_arr[i,j] = pfixth
    end
end
