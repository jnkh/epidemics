##################################################
######## A simple model of infection with variable timestep
##################################################


module IM

import Cubature
export p_birth,p_death,InfectionModel, plot_schematic, get_parameters


type InfectionModel
    birth_rate::Function
    death_rate::Function
    dt
end


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
    
    psi(x,a,b) = exp( -2* Cubature.hquadrature(y -> a(y)/b(y),0,x)[1])
    
    return Cubature.hquadrature(y -> psi(y,a,b),0,x0)[1]/Cubature.hquadrature(y -> psi(y,a,b),0,1)[1]
end

function P_fix(im::InfectionModel,N::Int,x0::Array)
    return [P_fix(im,N,xind) for xind in x0]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* Cubature.hquadrature(y -> a(y)/b(y),0,x)[1])
    
    return Cubature.hquadrature(y -> psi(y,a,b),0,x0)[1]/Cubature.hquadrature(y -> psi(y,a,b),0,x1)[1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
    return [P_reach(im,N,x0,xind) for xind in x1]
end




end
