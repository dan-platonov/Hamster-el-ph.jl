include(joinpath(@__DIR__,"MaterialPaths.jl"))
using .MaterialPaths
using HDF5
# Single-process pipeline stages; only accumulate_uH_parallel.jl uses MPI.
const STAGES = Dict(
    "covariance"=>"compute_equilibrium_covariance.jl",
    "d"=>"compute_d_from_accumulated_streaming.jl",
    "phonons"=>"compute_phonons_from_covariance.jl",
    "g-orbital"=>"compute_electron_phonon_g.jl",
    "unitary"=>"unitary_transformation_matrix_construction.jl",
    "g-band"=>"transform_electron_phonon_g_to_band_basis.jl")
if isempty(ARGS) || "--help" in ARGS || "-h" in ARGS
    println("Usage: ./run.sh STAGE --calc-dir INPUT_DIRECTORY [--data-dir OUTPUT_DIRECTORY] [stage options]")
    println("Stages: covariance, accumulate, d, phonons, g-orbital, unitary, g-band")
    println("Use MPI (-n > 1) only for accumulate. phonons requires --temperature T.")
    println("covariance: --trajectory FILE --repetitions N1 N2 N3 --start N --final N|all --no-overwrite; accumulate: --start N --final N --chunk-size N --correlations distributed|serial")
    println("Inputs: --calc-dir. Outputs: --data-dir, or CALC_DIR/data by default.")
    exit(0)
end
stage=popfirst!(ARGS)
haskey(STAGES,stage) || error("Unknown stage: $stage")
options=Dict{String,String}();remaining=String[]
i=1
while i<=length(ARGS)
    global i
    if ARGS[i] in ("--calc-dir","--data-dir")
        i<length(ARGS) || error("Missing value for $(ARGS[i])")
        options[ARGS[i]]=ARGS[i+1];i+=2
    else
        push!(remaining,ARGS[i]);i+=1
    end
end
haskey(options,"--calc-dir") || error("Supply --calc-dir INPUT_DIRECTORY")
material_root=MaterialPaths.material_directory
calc=abspath(expanduser(options["--calc-dir"]))
isdir(calc) || error("Missing input directory: $calc")
data=MaterialPaths.check_storage_path(get(options,"--data-dir",MaterialPaths.default_data_directory(calc)))
MaterialPaths.check_storage_path(data)
if stage == "phonons" && !("--pinv-atol" in remaining)
    regression_file = joinpath(data, "electron_lattice_D.h5")
    if isfile(regression_file)
        threshold = h5open(regression_file, "r") do f
            Float64(read(f["covariance_pseudoinverse/threshold_angstrom2"]))
        end
        isfinite(threshold) && threshold >= 0 || error("Invalid d-regression covariance threshold")
        # The q-local relative cutoff can otherwise retain nearly constrained
        # centre-of-mass modes that the full-covariance d regression excluded.
        append!(remaining, ["--pinv-atol", string(threshold)])
        println("Phonon covariance absolute cutoff from d regression: $threshold Angstrom^2")
    end
end
stage in ("d","unitary","g-band") && !isempty(remaining) && error("This stage accepts only --calc-dir and --data-dir")
ENV["EP_CALC_DIR"]=calc;ENV["EP_DATA_DIR"]=data;ENV["EP_RUN_DIR"]=calc
ENV["ELECTRON_PHONON_DATA_DIR"]=data
mkpath(data)
empty!(ARGS);append!(ARGS,remaining)
println("Material-independent pipeline; stage=$stage\nInputs: $calc\nData: $data")
Base.include(Main,joinpath(@__DIR__,STAGES[stage]))
