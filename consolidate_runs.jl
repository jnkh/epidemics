using EpidemicsSimulations

path = "/n/regal/desai_lab/juliankh/tmp/data/"
outpath = "~/physics/research/desai/epidemics/data/"
dirs = split(readall(`ls $path`));
for (i,curr_dir) in enumerate(dirs)
    consolidate_epidemic_runs(path * curr_dir,outpath)
    run(`rm -rf $(path)/$(curr_dir)`)
    println("$i/$(length(dirs))")
end

