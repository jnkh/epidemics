##################################################
######## A simple model of infection with variable timestep
##################################################


module IM

#import Cubature
export p_birth,p_death,InfectionModel, plot_schematic, get_parameters,P_reach,P_fix,P_reach_fast


type InfectionModel
    birth_rate::Function
    death_rate::Function
    dt
end

eps = 1e-6
max_evals = 200

function InfectionModel(birth_rate::Function,death_rate::Function)
    desired_rate = 0.1
    max_rate = maximum([birth_rate(1.0),death_rate(1.0)])
    dt = desired_rate/max_rate
    InfectionModel(birth_rate,death_rate,dt)
end

function p_birth(im::InfectionModel,x)
    return im.birth_rate(x)*im.dt
end

function p_death(im::InfectionModel,x)
    return im.death_rate(x)*im.dt
end

function s(im::InfectionModel,x)
    return (1- x)*(p_birth(im,x) - p_death(im,x))
end
    
function P_fix(im::InfectionModel,N::Int,x0::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x)[1])
    
    return quadgk(y -> psi(y,a,b),0,x0)[1]/quadgk(y -> psi(y,a,b),0,1)[1]
end

function P_fix(im::InfectionModel,N::Int,x0::Array)
    return [P_fix(im,N,xind) for xind in x0]
end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
#     s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
#     a(x) = x*s(x)
#     b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x)[1])
    
#     return quadgk(y -> psi(y,a,b),0,x0)[1]/quadgk(y -> psi(y,a,b),0,x1)[1]
# end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
#     return [P_reach(im,N,x0,xind) for xind in x1]
# end


function P_fix(s::Function,splus::Function,N::Int,x0::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,1,maxevals=max_evals)[1]
end


function P_fix(s::Function,splus::Function,N::Int,x0::Array)
    return [P_fix(s,splus,N,xind) for xind in x0]
end


function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,x1,maxevals=max_evals)[1]
end

function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    return [P_reach(s,splus,N,x0,xind) for xind in x1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x,maxevals=max_evals)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi(y,a,b),eps,x1,maxevals=max_evals)[1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
    return [P_reach(im,N,x0,xind) for xind in x1]
end

using Dierckx
function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    eps = 1e-5
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x,maxevals=max_evals)[1])
    xx = logspace(log10(1/N)-1,0,1000)
    yy = zeros(xx)
    for i in 1:length(xx)
        yy[i] = psi(xx[i],a,b)
    end
    psi_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
    psi_interp(x) = evaluate(psi_spline,x)
    
    return quadgk(y -> psi_interp(y),eps,x0,maxevals=max_evals)[1]/quadgk(y -> psi_interp(y),eps,x1,maxevals=max_evals)[1]
end

function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    return [P_reach_fast(s,splus,N,x0,xind) for xind in x1]
end




end
