using Base.Threads
using PayloadGraph

N = 100
m = Mutex()
p = PayloadGraph.Graph(LightGraphs.Graph(N,Int(round(N*(N-1)/2))),zeros(N))

matsize = 200
mat = randn(matsize,matsize)
println("Working with $(nthreads()) threads")

function set_graph_with_threads(p,m::Mutex)
  @threads all for i = 1:length(p.payload)
    payload = 1.0*mysum(mymatmul(mat,mat))
    lock!(m)
    set_payload(p,i,payload)
    unlock!(m)
  end
end

function set_graph_serial(p)
  for i = 1:length(p.payload)
    payload = 1.0*mysum(mymatmul(mat,mat))
    set_payload(p,i,payload)
  end
end
function mysum(arr::Array{Float64})
 cum = 0.0
 for i = 1:length(arr)
  cum += arr[i]
 end
 cum
end 

function mymatmul(arr1::Array{Float64,2},arr2::Array{Float64,2})
  assert(size(arr2)[1] == size(arr1)[2])
  c = Array(Float64,(size(arr1)[1],size(arr2)[2]))
  for i = 1:size(arr1)[1]
    for j = 1:size(arr2)[2]
      cum = 0.0
      for k = 1:size(arr2)[1]
        cum += arr1[i,k]*arr2[k,j]
      end
      c[i,j] = cum
    end
  end
  c
end

@time set_graph_serial(p)
p.payload *= 0.0 
@time set_graph_with_threads(p,m)   

println(p)

     
