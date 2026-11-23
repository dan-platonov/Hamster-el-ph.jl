include(joinpath(@__DIR__,"MaterialPaths.jl"))
using .MaterialPaths
# MPI-aware accumulation: one checkpoint owner, collective HAMSTER execution.
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
using MPI
import Hamster
using LinearAlgebra
BLAS.set_num_threads(1)
MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

function run_hamster_collective(filename)
    conf = Hamster.get_config(filename=filename)
    RANK != 0 && Hamster.set_value!(conf,"verbosity",0)
    started = time()
    Hamster.main(COMM,conf;rank=RANK,nranks=NRANKS,num_nodes=1)
    MPI.Barrier(COMM)
    if RANK == 0
        println("Parallel HAMSTER: ranks=$NRANKS, threads/rank=$(Threads.nthreads()), seconds=$(time()-started)")
        flush(stdout)
    end
end

# Workers service HAMSTER and distributed-correlation collectives.
function run_hamster_parallel(filename)
    MPI.bcast(filename,COMM;root=0)
    run_hamster_collective(filename)
end

function configure(args)
    options = Dict{String,String}()
    remaining = String[]
    i=1
    while i<=length(args)
        if args[i] in ("--calc-dir","--data-dir","--work-dir","--correlations")
            i<length(args) || error("Missing value for $(args[i])")
            options[args[i]]=args[i+1];i+=2
        else
            push!(remaining,args[i]);i+=1
        end
    end
    haskey(options,"--calc-dir") || error("Supply --calc-dir INPUT_DIRECTORY (relative to the current directory or absolute).")
    project_root=MaterialPaths.material_directory
    calc=abspath(expanduser(options["--calc-dir"]))
    isdir(calc) || error("Missing calculation directory: $calc")
    data=MaterialPaths.check_storage_path(get(options,"--data-dir",MaterialPaths.default_data_directory(calc)))
    work=MaterialPaths.check_storage_path(get(options,"--work-dir",data))
    for path in (data,work); MaterialPaths.check_storage_path(path); end
    if RANK==0
        mkpath(data);mkpath(work)
        if work!=calc
            for name in ("hconf","rllm.dat","params.dat","ml_params.dat","EIGENVAL","SC_POSCAR","POSCAR_primitive","POSCAR_equil_primitive","POSCAR_equlibr")
                source=joinpath(calc,name);destination=joinpath(work,name)
                isfile(source) && !ispath(destination) && symlink(source,destination)
            end
        end
    end
    MPI.Barrier(COMM)
    mode=get(options,"--correlations","distributed")
    mode in ("serial","distributed") || error("--correlations must be serial or distributed")
    ENV["EP_CORRELATIONS"]=mode
    ENV["EP_CALC_DIR"]=calc;ENV["EP_DATA_DIR"]=data;ENV["EP_RUN_DIR"]=work
    cd(work)
    empty!(ARGS);append!(ARGS,remaining)
    return calc,data,work
end

try
    if "--help" in ARGS && !("--calc-dir" in ARGS)
        RANK==0 && println("Usage: accumulate_uH_parallel.jl --calc-dir INPUT_DIRECTORY [--data-dir DIR] [--work-dir DIR] [--start N --final N --chunk-size N --skip-snapshots N --checkpoint-every-chunks N]\nLaunch with jrun -n RANKS -t THREADS using the supplied MPI launcher. Only rank 0 writes the accumulated checkpoint.")
    else
        calc,data,work=configure(copy(ARGS))
        distributed=ENV["EP_CORRELATIONS"]=="distributed"
        Base.include(Main,joinpath(@__DIR__,distributed ? "AccumulationCore.jl" : "AccumulationCore_serial.jl"))
        distributed && Base.include(Main,joinpath(@__DIR__,"DistributedCorrelations.jl"))
        if RANK==0
            mode_text=ENV["EP_CORRELATIONS"]; println("Correlation mode=$mode_text; ranks=$NRANKS; threads/rank=$(Threads.nthreads()); HDF5 writer=rank 0")
            println("Source: $calc\nData: $data\nWork: $work");flush(stdout)
            # The legacy --help handler exits the process; handle it collectively instead.
            help_only="--help" in ARGS || "-h" in ARGS
            if help_only
                Base.invokelatest(() -> getfield(Main,:print_help)())
            else
                Base.invokelatest(() -> getfield(Main,:main)())
            end
            MPI.bcast("",COMM;root=0)
        else
            while true
                message=MPI.bcast(nothing,COMM;root=0)
                if message isa AbstractString
                    isempty(message) && break
                    run_hamster_collective(message)
                else
                    Base.invokelatest(() -> handle_correlation_command(message))
                end
            end
        end
    end
    MPI.Barrier(COMM)
    MPI.Finalize()
catch err
    showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
    MPI.Abort(COMM,1)
end
