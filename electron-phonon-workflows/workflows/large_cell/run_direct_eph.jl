using TOML
if isempty(ARGS) || ARGS[1] in ("--help", "-h")
    println("Usage: julia --project=REPOSITORY run_direct_eph.jl CONFIG.toml")
    println("See examples/GaAs/direct_10x10x10.toml. Paths are relative to the TOML file.")
    exit(0)
end
length(ARGS)==1 || error("Supply exactly one TOML configuration")
const CONFIG_FILE=abspath(expanduser(ARGS[1]))
const CONFIG=TOML.parsefile(CONFIG_FILE)
const CONFIG_BASE=dirname(CONFIG_FILE)
resolve_path(value)=abspath(joinpath(CONFIG_BASE,expanduser(value)))
const CALC=resolve_path(CONFIG["calc_dir"])
const DATA=resolve_path(CONFIG["data_dir"])
mkpath(DATA)
include(joinpath(@__DIR__,"DirectFourierEph.jl"))
using .DirectFourierEph
final=get(CONFIG,"final_snapshot","all")
run_direct_eph(;
    xdatcar_file=resolve_path(CONFIG["trajectory"]),
    covariance_file=joinpath(DATA,"displacement_covariance.h5"),
    equilibrium_file=joinpath(CALC,"POSCAR_equlibr"),
    primitive_file=joinpath(CALC,"POSCAR_equil_primitive"),
    hconf_file=joinpath(CALC,"hconf"),
    output_file=joinpath(DATA,"electron_lattice_d_kq_direct.h5"),
    work_dir=joinpath(DATA,".direct_eph_work"),
    model_files=[joinpath(CALC,name) for name in CONFIG["model_files"]],
    start_snapshot=get(CONFIG,"start_snapshot",500),
    final_snapshot=final=="all" ? nothing : Int(final),
    skip_snapshots=get(CONFIG,"skip_snapshots",0),
    hamster_chunk_size=get(CONFIG,"hamster_chunk_size",20),
    accumulation_window=get(CONFIG,"accumulation_window",60),
    convergence_m=get(CONFIG,"convergence_m",1),
    convergence_log_file=joinpath(DATA,"d_convergence.txt"),
    diagnostic_q_block=get(CONFIG,"diagnostic_q_block",8),
    pseudoinverse_rtol=get(CONFIG,"pseudoinverse_rtol",1e-10),
    pseudoinverse_atol=get(CONFIG,"pseudoinverse_atol",0.0),
    h_tolerance=get(CONFIG,"h_tolerance",1e-14),
    fft_threads=get(CONFIG,"fft_threads",1),
    hamster_executable=get(CONFIG,"hamster_executable","hamster"),
    assemble_final_output=get(CONFIG,"assemble_final_output",false))
