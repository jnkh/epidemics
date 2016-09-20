push!(LOAD_PATH, pwd())
using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD, TwoLevelGraphs,Dierckx
import LightGraphs

#export plot_schematic,plot_schematics

function get_c_r(N,alpha,beta)
    return 4*alpha/(beta^2*N)
end

function get_n_n(N,alpha,beta)
    return beta/alpha*N
end

function get_alpha_beta(N,c_r,n_n)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
    return alpha,beta
end


f(y,alpha) = alpha.*y.^2
s(y,alpha,beta) = f(y,alpha)./y - beta
#get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
#get_s_eff(y::Array,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


function plot_schematics(N,n_n,alpha,beta,im,imk,k)
    plot_reach = true
    #pygui(true)
#     close("all")
    dx = 2*n_n/N/200
    x = collect(1/N:dx:2*n_n/N)
    if plot_reach
        y = IM.P_reach(im,N,1.0/N,x)
        yk = IM.P_reach(imk,N,1.0/N,x)
        plotfn = loglog
        plotstr = "reach"
    else
        y = IM.P_fix(im,N,x)
        yk = IM.P_fix(imk,N,x)
        plotfn = plot
        plotstr = "fix"
    end
    figure(2)#,figsize=(8,5))
    plotfn(x,y,"-r",label=latexstring("P_{$(plotstr)}(y)"))
    plotfn(x,yk,"-b",label=latexstring("P_{$(plotstr)}(y_{eff})"))
    plotfn(x,1/N./x,"--k",label=latexstring("P_{$(plotstr),neutral}"))
    xlim([1/N,2*n_n/N])
    y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta)
    axvline(y_n,linestyle="--",color="b",label=L"y_n")
    axvline(y_minus,linestyle="-.",color="r",label=L"y_1")
    axvline(y_plus,linestyle="-.",color="r",label=L"y_2")
    axvline(y_p,linestyle="-",color="b",label=L"y_p")
    xlabel(L"y")
    ylabel(latexstring("P_{$(plotstr)}(y)"))
    legend(loc="best")
    title(latexstring("\$y_n = $(n_n/N), c_r = $c_r, N = $N\$"))
    #savefig("p_fix_y_n = $(n_n/N), c_r = $c_r, N = $N.png")

    figure(1)#,figsize=(8,5))
    plot_schematic(n_n,c_r,N,k,true)
    title(latexstring("\$y_n = $(n_n/N), c_r = $c_r, N = $N\$"))
    
    x = collect(1/N:0.01:1)
    figure(3)
    plot(x,get_s_eff(x,alpha,beta,k),"-b",label=L"$s_{eff}(y)$")
    plot(x,get_s_eff(x,alpha,beta,N-1),"-r",label=L"$s(y)$")
    grid(1)

end


function plot_schematic(n_n,c_r,N,k=N-1,plot_k=false)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
#     println(N,alpha,beta)

    y_n, y_minus,y_plus,y_p,critical_determinant = get_parameters(N,alpha,beta)
    f(y) = alpha.*y.^2
    s(y) = f(y)./y - beta
    get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
    get_s_eff(y,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


    y_range = collect(0:y_p/1000:1.9*y_p)
    plot(y_range,1.0./abs(N*s(y_range)),"-r",label=L"$\frac{1}{N|s(y)|}$")
    if plot_k
        plot(y_range,1.0./abs(N*get_s_eff(y_range,alpha,beta,k)),"-b",label=L"$\frac{1}{N|s(y_{eff})|}$")
    end
    plot(y_range,y_range,"-k",label=L"$y$")
    axvline(y_n,linestyle="--",label=L"$y_n$")
    axvline(y_p,linestyle="-",label=L"$y_p$")
    if y_minus > 0
        axvline(y_minus,linestyle="-.",label=L"$y_1$")
        axvline(y_plus,linestyle="-.",label=L"$y_2$")
    end
    ylim([0,1.9*y_p])
    legend(prop=Dict{Any,Any}("size"=>15),loc="upper right")
    xlabel(L"$y$",size=20)
    if plot_k
        #figure(1)
        #plot(y_range,get_s_eff(y_range,alpha,beta,k),"-b",label=L"$s_{eff}(y)$")
        #plot(y_range,s(y_range),"-r",label=L"$s(y)$")
        legend(prop=Dict{Any,Any}("size"=>20),loc="upper right")
        xlabel(L"$y$",size=20)
    end
end

function plot_w(sizes,N,alpha,beta,k::Int,word = "two level")

    figure(4)
    sizes = (1 + beta).*sizes
    bins = logspace(log10(minimum(sizes)),log10(maximum(sizes)),150)
#     word = graph_model ? "graph" : "well-mixed"
    label = latexstring("$word, \$k = $k\$")
    PyPlot.plt[:hist](sizes,log=true,bins=bins,alpha=0.2,normed=true,label=label)

    gca()[:set_xscale]("log")

    w_range = bins[1:end]#logspace(log10(4*minimum(sizes)),log10(maximum(sizes)),30)

    P_w_th_range = normed_distribution(w_range,P_w_th(w_range,s(sqrt(w_range)./N,alpha,beta)))
    P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,get_s_eff(sqrt(w_range)./N,alpha,beta,k)))
    #P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,
    #get_s_effective_two_level_interp(sqrt(w_range)./N,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)))
    
    correction_fac = 5
    #plot(w_range,correction_fac*P_w_th_range,"-r",label=L"theory $k \to N-1$")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    if true#graph_model
        plot(w_range,correction_fac*P_w_th_range_eff,"-b",label=latexstring("effective theory \$k = $k\$"))#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    end
        
    xlabel(L"$w$",size=20)
    ylabel(L"$P(w)$",size=20)

    legend(loc="lower left")
    ylim([1e-6,1e3])
    grid()

end
function plot_w(sizes,N,alpha,beta,s_eff_fn::Function, word = "two level")

    figure(4)
    sizes = (1 + beta).*sizes
    bins = logspace(log10(minimum(sizes)),log10(maximum(sizes)),150)
#     word = graph_model ? "graph" : "well-mixed"
    label = latexstring("$word, \$k = $k\$")
    PyPlot.plt[:hist](sizes,log=true,bins=bins,alpha=0.2,normed=true,label=label)

    gca()[:set_xscale]("log")

    w_range = bins[1:end]#logspace(log10(4*minimum(sizes)),log10(maximum(sizes)),30)

    P_w_th_range = normed_distribution(w_range,P_w_th(w_range,s(sqrt(w_range)./N,alpha,beta)))
    #P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,get_s_eff(sqrt(w_range)./N,alpha,beta,k)))
    P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,
    s_eff_fn(sqrt(w_range))./N))
    
    correction_fac = 5
    #plot(w_range,correction_fac*P_w_th_range,"-r",label=L"theory $k \to N-1$")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    if true#graph_model
        plot(w_range,correction_fac*P_w_th_range_eff,"-g",label="two-level eff. theory")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    end
        
    xlabel(L"$w$",size=20)
    ylabel(L"$P(w)$",size=20)

    legend(loc="lower left")
    ylim([1e-6,1e3])
    grid()

end

# s(x) = get_s_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
# splus(x) = get_splus_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)

# s1(x) = alpha*x - beta
# splus1(x) = 2 + beta + alpha*x


# s2(x) = p_birth(im_effective,x) - p_death(im_effective,x)
# splus2(x) =  p_birth(im_effective,x) + p_death(im_effective,x)

# max_evals = 1000

# function P_fix(s::Function,splus::Function,N::Int,x0::Real)
#     eps = 1e-5
#     a(x) = s(x)
#     b(x) = 1/N*(splus(x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
    
#     return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,1,maxevals=max_evals)[1]
# end
# import IM.P_reach
# function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
#     eps = 1e-5
#     a(x) = s(x)
#     b(x) = 1/N*(splus(x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
    
#     return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,x1,maxevals=max_evals)[1]
# end

# function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
#     return [P_reach(s,splus,N,x0,xind) for xind in x1]
# end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
#     eps = 1e-6
#     s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
#     a(x) = x*s(x)
#     b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x,maxevals=max_evals)[1])
    
#     return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,x1,maxevals=max_evals)[1]
# end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
#     return [P_reach(im,N,x0,xind) for xind in x1]
# end

# using Dierckx
# function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
#     eps = 1e-5
#     a(x) = s(x)
#     b(x) = 1/N*(splus(x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
#     xx = logspace(log10(1/N)-1,0,1000)
#     yy = zeros(xx)
#     for i in 1:length(xx)
#         yy[i] = psi(xx[i],a,b)
#     end
#     psi_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
#     psi_interp(x) = evaluate(psi_spline,x)
    
#     return quadgk(y -> psi_interp(y),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi_interp(y),eps,x1,maxevals=max_evals)[1]
# end

# function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
#     return [P_reach_fast(s,splus,N,x0,xind) for xind in x1]
# end



function plot_two_level_schematic(t,alpha,beta,N)
    y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp = get_interpolations(t,alpha,beta)
    println("computed interpolations")

    s(x) = get_s_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
    splus(x) = get_splus_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)

    xx = logspace(log10(1/N),0,100) 
    pp = P_reach_fast(s,splus,N,1/N,xx)
#     loglog(xx,pp)
#     xlim([1/N,1])
    return xx,pp,s
end

k = 12
#y_n = 0.1
c_r = 0.2 #0.18
N = 400#100000#400
n_n = 200#10#80#y_n*N
beta = 4.0/(c_r*n_n)
alpha = (N*beta)/n_n
println("N=$N, alpha = $alpha, beta = $beta")

#Generate a random startin vector
m = 4#20 number of nodes in a community
n = Int(N/m) 
l = 2#Int(m/2)#10#internal
r = 10#2#Int(m/2)#2 #external

im = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
imk = InfectionModel(x -> 1 + beta + get_s_eff(x,alpha,beta,k) , x -> 1 + beta);


y_desired = 0.003

t = TwoLevel(N,m,l,r)
#distribute_randomly(t,n)
adjust_infecteds(t,y_desired)
make_consistent(t)
assert(is_valid(t))
println(t.i/t.N)
xx,yy,s_eff_two_level = plot_two_level_schematic(t,alpha,beta,N)