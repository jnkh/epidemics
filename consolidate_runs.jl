using EpidemicsSimulations

path = "/n/regal/desai_lab/juliankh/tmp/"
outpath = "/n/home07/juliankh/physics/research/desai/epidemics/data/"
dirs = split(readall(`ls $path`));
for (i,curr_dir) in enumerate(dirs)
    consolidate_epidemic_runs(path * curr_dir * "/data/",outpath)
    run(`rm -rf $(path)/$(curr_dir)`)
    println("$i/$(length(dirs))")
end

