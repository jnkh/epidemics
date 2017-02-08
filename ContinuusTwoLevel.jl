module ContinuusTwoLevel

using PyPlot,IM, Dierckx, TwoLevelGraphs, Epidemics

export get_a_distr,get_a_b,get_exact_interpolations

function get_a_distr(a::Function,b::Function,num_points)
    psi1 = IM.get_psi_interp(x -> -1.0*a(x),b,1e-5,0,num_points)
    f(z) = psi1(z)./b(z)
    zz = 0:1/num_points:1
    distr = f(zz)
    distr /= sum(distr)
    spline = Spline1D(zz,distr,k=1,bc="extrapolate")
    distr_fn(x) = evaluate(spline,x)
    return distr_fn
end

function get_a_b(y,delta,m)
    a(z) = (1-delta)*(y-z)
    b(z) = 1/m*(2*delta*z.*(1-z) + (1-delta)*(y*(1-z) + z*(1-y)))
    return a,b
end

function get_y_local(delta,y,z)
    return delta*z + (1 - delta)*y
end

function get_y_sq_local(delta,y,z)
    return delta*z.^2 + (1 - delta)*y.^2
end

function get_number_weight(distr,z,susc=true)
    arr = 0
    if susc
        arr = distr(z).*(1-z)
    else
        arr = distr(z).*z
    end
    return arr/sum(arr) 
end


function get_mean_y_local(y,delta,m,susc=true,num_points = 1000)
    a,b = get_a_b(y,delta,m)
    distr_fn = get_a_distr(a,b,num_points)
    zz = 0:1/num_points:1
    return sum(get_number_weight(distr_fn,zz,susc).*get_y_local(delta,y,zz))
end

function get_mean_y_sq_local(y,delta,m,susc=true,num_points = 1000)
    a,b = get_a_b(y,delta,m)
    distr_fn = get_a_distr(a,b,num_points)
    zz = 0:1/num_points:1
    return sum(get_number_weight(distr_fn,zz,susc).*get_y_sq_local(delta,y,zz))
end   


function get_exact_interpolations(delta,m,alpha,beta,num_points = 200)

    eps = 1e-5
    yy = collect(eps:1/num_points:1-eps)
    interpolation_order = 1

    yy_sc = [get_mean_y_local(_,delta,m,true) for _ in yy]
    yy_inf = [get_mean_y_local(_,delta,m,false) for _ in yy]
    yy_sc_sq = [get_mean_y_sq_local(_,delta,m,true) for _ in yy]
    yy_inf_sq = [get_mean_y_sq_local(_,delta,m,false) for _ in yy]

    s_eff_range = TwoLevelGraphs.get_s_effective_two_level(yy,yy_sc,yy_sc_sq,yy_inf,yy_inf_sq,alpha,beta)
    splus_eff_range = TwoLevelGraphs.get_splus_effective_two_level(yy,yy_sc,yy_sc_sq,yy_inf,yy_inf_sq,alpha,beta)
    s_birth_range = TwoLevelGraphs.get_s_birth_effective_two_level(yy,yy_sc,yy_sc_sq,alpha) 
    s_death_range = TwoLevelGraphs.get_s_death_effective_two_level(yy,yy_inf,beta)

    y_inf_interp = Spline1D(yy,yy_inf,k=interpolation_order,bc="extrapolate")
    y_susc_interp = Spline1D(yy,yy_sc,k=interpolation_order,bc="extrapolate")
    y_sq_inf_interp = Spline1D(yy,yy_inf_sq,k=interpolation_order,bc="extrapolate")
    y_sq_susc_interp = Spline1D(yy,yy_sc_sq,k=interpolation_order,bc="extrapolate")

    s_birth_interp = Spline1D(yy,s_birth_range,k=interpolation_order,bc="extrapolate")
    s_death_interp = Spline1D(yy,s_death_range,k=interpolation_order,bc="extrapolate")
    s_interp = Spline1D(yy,s_eff_range,k=interpolation_order,bc="extrapolate")
    splus_interp = Spline1D(yy,splus_eff_range,k=interpolation_order,bc="extrapolate")
    
    return y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp,s_birth_interp,s_death_interp,s_interp,splus_interp
end


end