module CalculationPaths

export resolve_calculation_directory, stage_data_directory!, default_mass_file,
       selected_covariance_file, directory_temperature

const PROJECT_ROOT = realpath(joinpath(@__DIR__, ".."))
const DATA_ROOT = get(ENV, "EPH_DATA_ROOT", PROJECT_ROOT)

"""Accept a data directory or its matching Ga_As source directory."""
function resolve_calculation_directory(directory::AbstractString)
    path = realpath(expanduser(directory))
    isdir(path) || error("Not a directory: $directory")
    nested = joinpath(path, "data")
    if isdir(nested) && (isfile(joinpath(nested, "electron_lattice_d_kq.h5")) ||
                        isfile(joinpath(nested, ".direct_eph_work", "pair_0001_0001.h5")))
        return realpath(nested)
    end
    isfile(joinpath(path, "electron_lattice_d_kq.h5")) && return path
    relative = relpath(path, PROJECT_ROOT)
    parts = splitpath(relative)
    if length(parts) == 3 && first(parts) != ".."
        mirrored = joinpath(DATA_ROOT, relative)
        isdir(mirrored) && return realpath(mirrored)
    end
    return path
end

"""Consume --directory DIR; keep the existing stage-specific CLI arguments."""
function stage_data_directory!(args::Vector{String}; path_only::Bool=false)
    directory = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--directory"
            isnothing(directory) || error("Specify --directory only once.")
            i < length(args) || error("Missing value after --directory")
            directory = args[i+1]
            deleteat!(args, i:i+1)
        else
            i += 1
        end
    end
    if any(arg -> arg in ("--help", "-h"), args)
        println("Shared input/output directory: --directory DIR (a data or Ga_As calculation directory).")
        if path_only
            println("Usage: julia $(basename(PROGRAM_FILE)) --directory DIR")
            exit(0)
        end
        return pwd()
    end
    if isnothing(directory)
        directory = get(ENV, "ELECTRON_PHONON_DATA_DIR", pwd())
    end
    path_only && !isempty(args) && error("Unknown arguments: $(join(args, ' ')); use --help.")
    return resolve_calculation_directory(directory)
end

function default_mass_file(directory::AbstractString)
    local_file = joinpath(directory, "atomic_masses_amu.txt")
    return isfile(local_file) ? local_file : joinpath(@__DIR__, "atomic_masses_amu.txt")
end

function selected_covariance_file(directory::AbstractString)
    aligned = joinpath(directory, "displacement_covariance_hamiltonian_labels.h5")
    return isfile(aligned) ? aligned : joinpath(directory, "displacement_covariance.h5")
end

function directory_temperature(directory::AbstractString)
    matched = match(r"^(\d+(?:\.\d+)?)K$"i, basename(directory))
    isnothing(matched) && return nothing
    return parse(Float64, matched.captures[1])
end

end
