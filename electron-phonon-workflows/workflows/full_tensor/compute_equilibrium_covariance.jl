####nobackup directory
const CALC_DIR = ENV["EP_CALC_DIR"]
const NOBACKUP_DIR = ENV["EP_DATA_DIR"]
mkpath(NOBACKUP_DIR)
####

using HDF5
using LinearAlgebra
using Printf
# Covariance BLAS work uses the threads reserved for this single-rank stage.
BLAS.set_num_threads(Threads.nthreads())
include(joinpath(@__DIR__, "MaceTrajectory.jl"))
const MACE_TIMESTEPS = Dict{Int,Int}()
is_mace_file(path) = open(io -> startswith(readline(io), "ITEM: TIMESTEP"), path, "r")

# =============================================================================
# Streaming finite-temperature equilibrium + covariance from native MACE MD
#
# Inputs:
#   SC_XDATCAR
#   POSCAR_primitive
#
# Outputs:
#   POSCAR_equlibr
#   POSCAR_equil_primitive
#   displacement_covariance.h5
#
# No HAMSTER Hamiltonian file is used in this stage.
#
# Snapshot numbers below are trajectory frame ordinals (1-based), not necessarily
# the integer printed after "Direct configuration=".
#
# Examples:
#   julia compute_equilibrium_covariance_streaming.jl
#   julia compute_equilibrium_covariance_streaming.jl --start 500 --final 5000
#   julia compute_equilibrium_covariance_streaming.jl --start 1000 --final all
# =============================================================================

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------



# Heavy files
const DEFAULT_XDATCAR_FILE =
    (isfile(joinpath(NOBACKUP_DIR,"positions.lammpstrj")) ? joinpath(NOBACKUP_DIR,"positions.lammpstrj") : (isfile(joinpath(CALC_DIR,"positions.lammpstrj")) ? joinpath(CALC_DIR,"positions.lammpstrj") : (isfile(joinpath(NOBACKUP_DIR,"SC_XDATCAR")) ? joinpath(NOBACKUP_DIR,"SC_XDATCAR") : joinpath(CALC_DIR,"SC_XDATCAR"))))

const DEFAULT_COVARIANCE_OUTPUT =
    joinpath(NOBACKUP_DIR, "displacement_covariance.h5")

# Inputs are used in place; no implicit relocation of large files.
# Small files remain beside the script.
const DEFAULT_PRIMITIVE_POSCAR =
    (isfile(joinpath(CALC_DIR,"POSCAR_primitive")) ? joinpath(CALC_DIR,"POSCAR_primitive") : joinpath(NOBACKUP_DIR,"POSCAR_primitive"))

const DEFAULT_EQUILIBRIUM_OUTPUT =
    joinpath(NOBACKUP_DIR, "POSCAR_equlibr")

const DEFAULT_EQUILIBRIUM_PRIMITIVE_OUTPUT =
    joinpath(NOBACKUP_DIR, "POSCAR_equil_primitive")

# Infer the topology repetitions from the selected calculation directory.
const CELL_REPETITIONS_MATCH = match(r"(?:^|/)(\d+)x(\d+)x(\d+)(?:/|$)", abspath(CALC_DIR))
const DEFAULT_REPETITIONS = CELL_REPETITIONS_MATCH === nothing ? (4,4,4) :
    Tuple(parse.(Int,CELL_REPETITIONS_MATCH.captures))
const DEFAULT_START_SNAPSHOT = 500
const DEFAULT_FINAL_SNAPSHOT = nothing   # nothing = use all remaining frames
# Console-output controls
# Julia uses lowercase `true` / `false`.
const SHOW_PROGRESS = true
const PROGRESS_EVERY = 500

# Normal status output goes through this helper. When SHOW_PROGRESS=false,
# the script produces no console status/progress/final-summary output.
progress_println(args...) = if SHOW_PROGRESS; println(args...); flush(stdout); end

const LATTICE_DIRECTION_TOLERANCE = 5.0e-3
const POSITION_ASSIGNMENT_TOLERANCE_ANGSTROM = 0.75
# CsPbBr3 425K: verified unique mapping, maximum mean-site residual 0.29277 A.
# This acceptance check does not change the translational projection or estimator.
const INTEGER_CELL_TOLERANCE_ANGSTROM = 0.35

# -----------------------------------------------------------------------------
# Small data structures
# -----------------------------------------------------------------------------

struct StructureHeader
    comment::String
    lattice::Matrix{Float64}       # 3x3, lattice vectors are columns
    species_names::Vector{String}
    species_counts::Vector{Int}
    atom_species::Vector{String}
end

struct Structure
    header::StructureHeader
    fractional_positions::Matrix{Float64}  # 3 x Natoms
end

struct AtomPrimitiveMapping
    repetitions::NTuple{3,Int}
    number_of_primitive_cells::Int
    number_of_primitive_atoms::Int
    number_of_supercell_atoms::Int
    atom_to_primitive_atom::Vector{Int}
    atom_to_cell::Vector{NTuple{3,Int}}
    equilibrium_primitive_fractional::Matrix{Float64}
    equilibrium_primitive_lattice::Matrix{Float64}
end

# -----------------------------------------------------------------------------
# Command-line parsing
# -----------------------------------------------------------------------------

function print_help()
    println("""
Usage:
    julia compute_equilibrium_covariance_streaming.jl [options]

Options:
    --start N
        First trajectory frame ordinal to use (1-based).
        Default: $(DEFAULT_START_SNAPSHOT)

    --final N
        Last trajectory frame ordinal to use, inclusive.
        Default: all frames from --start to end of XDATCAR.

    --final all
        Explicitly request all remaining frames.

    --trajectory FILE
        Native MACE/LAMMPS positions.lammpstrj (legacy --xdatcar is accepted).
        Default: $(DEFAULT_XDATCAR_FILE)

    --repetitions N1 N2 N3
        Supercell repetitions used to extract primitive sites from frame 1. Default: inferred from the calculation directory ($(DEFAULT_REPETITIONS)).

    --primitive FILE
        Optional explicit reference primitive POSCAR; otherwise generated from native frame 1.
        Reference primitive POSCAR used only to define primitive-cell
        topology, species labels, and the primitive atom labels alpha.
        Its internal coordinates are NOT imposed on the finite-T equilibrium.
        Default: $(DEFAULT_PRIMITIVE_POSCAR)

    --equilibrium-output FILE
        Symmetrized finite-temperature supercell POSCAR.
        Default: $(DEFAULT_EQUILIBRIUM_OUTPUT)

    --primitive-equilibrium-output FILE
        Finite-temperature primitive POSCAR extracted from the symmetrized
        supercell equilibrium.
        Default: $(DEFAULT_EQUILIBRIUM_PRIMITIVE_OUTPUT)

    --covariance-output FILE
        HDF5 file containing the displacement covariance and labels.
        Default: $(DEFAULT_COVARIANCE_OUTPUT)


    --projection-tolerance ANGSTROM
        Maximum accepted residual before primitive-symmetry projection; default 0.35.

    --reuse-xdatcar
        Reuse a completed export from this same native trajectory on a preparation rerun.

    --no-overwrite
        Refuse to overwrite existing output files.

    --help
        Show this message.

Important:
    Fixed-cell native MACE dumps are read directly; XDATCAR remains supported.
    SC_POSCAR, POSCAR_primitive, equilibrium POSCARs, covariance, and a full
    SC_XDATCAR for the later HAMSTER stage are written under the selected data directory.
    --start/--final are 1-based frame ordinals; frame 1 has LAMMPS step 0.
""")
end

function parse_arguments(args)
    settings = Dict{Symbol,Any}(
        :start => DEFAULT_START_SNAPSHOT,
        :final => DEFAULT_FINAL_SNAPSHOT,
        :xdatcar => DEFAULT_XDATCAR_FILE,
        :primitive => DEFAULT_PRIMITIVE_POSCAR,
        :equilibrium_output => DEFAULT_EQUILIBRIUM_OUTPUT,
        :primitive_equilibrium_output => DEFAULT_EQUILIBRIUM_PRIMITIVE_OUTPUT,
        :covariance_output => DEFAULT_COVARIANCE_OUTPUT,
        :overwrite => true,
        :repetitions => DEFAULT_REPETITIONS,
        :projection_tolerance => INTEGER_CELL_TOLERANCE_ANGSTROM,
        :reuse_xdatcar => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)

        elseif arg == "--reuse-xdatcar"
            settings[:reuse_xdatcar] = true
            i += 1
            continue
        elseif arg == "--projection-tolerance"
            i < length(args) || error("Missing projection tolerance")
            settings[:projection_tolerance] = parse(Float64,args[i+1])
            isfinite(settings[:projection_tolerance]) && settings[:projection_tolerance] > 0 || error("Projection tolerance must be finite and positive")
            i += 2
            continue
        elseif arg == "--repetitions"
            i+3 <= length(args) || error("--repetitions needs N1 N2 N3")
            settings[:repetitions] = Tuple(parse.(Int,args[i+1:i+3]))
            all(>(0),settings[:repetitions]) || error("Repetitions must be positive")
            i += 4
            continue
        elseif arg == "--no-overwrite"
            settings[:overwrite] = false
            i += 1
            continue

        elseif arg in (
            "--start",
            "--final",
            "--xdatcar",
            "--trajectory",
            "--primitive",
            "--equilibrium-output",
            "--primitive-equilibrium-output",
            "--covariance-output",
        )
            i == length(args) && error("Missing value after $arg")
            value = args[i + 1]

            if arg == "--start"
                settings[:start] = parse(Int, value)
            elseif arg == "--final"
                settings[:final] =
                    lowercase(value) == "all" ? nothing : parse(Int, value)
            elseif arg in ("--xdatcar", "--trajectory")
                settings[:xdatcar] = abspath(value)
            elseif arg == "--primitive"
                settings[:primitive] = abspath(value)
            elseif arg == "--equilibrium-output"
                settings[:equilibrium_output] = abspath(value)
            elseif arg == "--primitive-equilibrium-output"
                settings[:primitive_equilibrium_output] = abspath(value)
            elseif arg == "--covariance-output"
                settings[:covariance_output] = abspath(value)
            end

            i += 2
            continue
        else
            error("Unknown command-line argument: $arg. Use --help.")
        end
    end

    settings[:start] >= 1 || error("--start must be >= 1.")
    if !isnothing(settings[:final])
        settings[:final] >= settings[:start] ||
            error("--final must be >= --start.")
    end

    return settings
end

# -----------------------------------------------------------------------------
# POSCAR / XDATCAR parsing
# -----------------------------------------------------------------------------

function expanded_species(names::Vector{String}, counts::Vector{Int})
    result = String[]
    for (name, count) in zip(names, counts)
        append!(result, fill(name, count))
    end
    return result
end

function parse_lattice(comment::AbstractString, scale_line::AbstractString, lattice_lines::AbstractVector{<:AbstractString})
    scale_values = parse.(Float64, split(strip(scale_line)))
    length(scale_values) in (1, 3) ||
        error("Expected one or three POSCAR scaling values.")

    raw = zeros(Float64, 3, 3)
    for i in 1:3
        values = parse.(Float64, split(strip(lattice_lines[i])))
        length(values) >= 3 || error("Invalid lattice-vector line.")
        raw[:, i] .= values[1:3]
    end

    if length(scale_values) == 1
        s = only(scale_values)
        if s > 0
            return s .* raw
        elseif s < 0
            target_volume = abs(s)
            raw_volume = abs(det(raw))
            raw_volume > 0 || error("Singular lattice in POSCAR/XDATCAR.")
            factor = (target_volume / raw_volume)^(1 / 3)
            return factor .* raw
        else
            error("POSCAR/XDATCAR scaling factor cannot be zero.")
        end
    else
        all(>(0), scale_values) ||
            error("Three-component lattice scaling factors must be positive.")
        scaled = copy(raw)
        for cart in 1:3
            scaled[cart, :] .*= scale_values[cart]
        end
        return scaled
    end
end

function read_poscar(filename::AbstractString)
    isfile(filename) || error("POSCAR file does not exist: $filename")

    open(filename, "r") do io
        comment = strip(readline(io))
        scale_line = readline(io)
        lattice_lines = [readline(io) for _ in 1:3]
        lattice = parse_lattice(comment, scale_line, lattice_lines)

        species_line = split(strip(readline(io)))
        counts_line = split(strip(readline(io)))

        # VASP 5+ format is expected here.
        all(token -> tryparse(Int, token) !== nothing, species_line) &&
            error(
                "VASP-4-style POSCAR without species names is not supported. " *
                "Please use a POSCAR containing explicit species names."
            )

        species_names = String.(species_line)
        species_counts = parse.(Int, counts_line)

        length(species_names) == length(species_counts) ||
            error("Species names and atom counts have different lengths.")

        natoms = sum(species_counts)
        atom_species = expanded_species(species_names, species_counts)

        line = strip(readline(io))
        if startswith(lowercase(line), "s")
            line = strip(readline(io))
        end

        lower = lowercase(line)
        direct = startswith(lower, "d")
        cartesian = startswith(lower, "c") || startswith(lower, "k")
        (direct || cartesian) ||
            error("Expected Direct or Cartesian coordinate mode in $filename.")

        coordinates = zeros(Float64, 3, natoms)
        for atom in 1:natoms
            values = parse.(Float64, split(strip(readline(io))))
            length(values) >= 3 ||
                error("Invalid coordinate line for atom $atom in $filename.")
            coordinates[:, atom] .= values[1:3]
        end

        fractional = direct ? coordinates : lattice \ coordinates
        fractional .= mod.(fractional, 1.0)

        header = StructureHeader(
            comment,
            lattice,
            species_names,
            species_counts,
            atom_species,
        )

        return Structure(header, fractional)
    end
end

function read_xdatcar_header(io)
    comment = strip(readline(io))
    scale_line = readline(io)
    lattice_lines = [readline(io) for _ in 1:3]
    lattice = parse_lattice(comment, scale_line, lattice_lines)

    species_names = String.(split(strip(readline(io))))
    species_counts = parse.(Int, split(strip(readline(io))))

    length(species_names) == length(species_counts) ||
        error("XDATCAR species names and atom counts have different lengths.")

    atom_species = expanded_species(species_names, species_counts)

    return StructureHeader(
        comment,
        lattice,
        species_names,
        species_counts,
        atom_species,
    )
end

function parse_marker_label(line::AbstractString, fallback::Int)
    m = match(r"configuration\s*=\s*([+-]?\d+)"i, line)
    return isnothing(m) ? fallback : parse(Int, m.captures[1])
end

"""
    foreach_xdatcar_frame(callback, filename, start_index, final_index)

Stream selected fixed-cell trajectory frames. Frame indices are 1-based ordinal
positions in the XDATCAR. `final_index === nothing` means to EOF.

The callback receives
    callback(frame_index, marker_label, fractional_positions, header)
"""
function foreach_xdatcar_frame(
    callback,
    filename::AbstractString,
    start_index::Int,
    final_index::Union{Nothing,Int},
)
    isfile(filename) || error("Trajectory does not exist: $filename")
    if is_mace_file(filename)
        return MaceTrajectory.foreach_frame(filename,start_index,final_index) do ordinal,frame
            MACE_TIMESTEPS[ordinal] = frame.step
            header = StructureHeader("CsPbBr3 MACE", frame.lattice, frame.species,
                                     frame.counts, frame.atom_species)
            callback(ordinal, ordinal, frame.fractional, header)
        end
    end

    selected_count = 0
    last_frame_seen = 0

    open(filename, "r") do io
        header = read_xdatcar_header(io)
        natoms = length(header.atom_species)

        frame_index = 0

        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue

            lower = lowercase(line)

            if startswith(lower, "direct configuration")
                frame_index += 1
                marker_label = parse_marker_label(line, frame_index)

                positions = zeros(Float64, 3, natoms)
                for atom in 1:natoms
                    eof(io) &&
                        error("Unexpected EOF while reading trajectory frame $frame_index.")
                    values = parse.(Float64, split(strip(readline(io))))
                    length(values) >= 3 ||
                        error(
                            "Invalid coordinate line in trajectory frame $frame_index, " *
                            "atom $atom."
                        )
                    positions[:, atom] .= values[1:3]
                end

                last_frame_seen = frame_index

                if frame_index >= start_index &&
                   (isnothing(final_index) || frame_index <= final_index)
                    callback(frame_index, marker_label, positions, header)
                    selected_count += 1
                end

                if !isnothing(final_index) && frame_index >= final_index
                    break
                end

            elseif startswith(lower, "cartesian configuration")
                error(
                    "Cartesian trajectory frames are not supported by this script. " *
                    "A standard fixed-cell VASP XDATCAR should contain Direct " *
                    "configuration blocks."
                )
            else
                # Ignore blank/comment-like lines between frames. A variable-cell
                # XDATCAR that repeats full headers is intentionally not supported.
            end
        end
    end

    return selected_count, last_frame_seen
end

# -----------------------------------------------------------------------------
# Periodic geometry helpers
# -----------------------------------------------------------------------------

function nearest_periodic_delta_fractional(
    delta_fractional::AbstractVector{<:Real},
    lattice::AbstractMatrix{<:Real},
)
    # For the diagonal fixed MACE box, closest images separate by component.
    # ceil(x-1/2) also preserves the original loop's positive-image tie choice.
    if isdiag(lattice)
        delta = Float64.(delta_fractional)
        return delta .- ceil.(delta .- 0.5)
    end
    base = round.(Int, delta_fractional)
    best_delta = zeros(Float64, 3)
    best_norm2 = Inf

    for i in -1:1, j in -1:1, k in -1:1
        shift = base .+ (i, j, k)
        candidate = Float64.(delta_fractional) .- collect(shift)
        cart = lattice * candidate
        norm2 = dot(cart, cart)

        if norm2 < best_norm2
            best_norm2 = norm2
            best_delta .= candidate
        end
    end

    return best_delta
end

function minimum_image_displacements(
    fractional_positions::Matrix{Float64},
    reference_fractional_positions::Matrix{Float64},
    lattice::Matrix{Float64},
)
    size(fractional_positions) == size(reference_fractional_positions) ||
        error("Position arrays have different dimensions.")

    natoms = size(fractional_positions, 2)
    displacements = zeros(Float64, 3, natoms)

    for atom in 1:natoms
        delta_frac =
            fractional_positions[:, atom] .-
            reference_fractional_positions[:, atom]

        wrapped_delta =
            nearest_periodic_delta_fractional(delta_frac, lattice)

        displacements[:, atom] .= lattice * wrapped_delta
    end

    return displacements
end

function nearest_integer_translation(
    x_minus_tau::AbstractVector{<:Real},
    lattice::Matrix{Float64},
)
    base = round.(Int, x_minus_tau)
    best = collect(base)
    best_norm2 = Inf

    for i in -1:1, j in -1:1, k in -1:1
        candidate = collect(base .+ (i, j, k))
        residual = Float64.(x_minus_tau) .- candidate
        cart = lattice * residual
        norm2 = dot(cart, cart)

        if norm2 < best_norm2
            best_norm2 = norm2
            best .= candidate
        end
    end

    return (best[1], best[2], best[3]), sqrt(best_norm2)
end

function periodic_distance(
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    lattice::Matrix{Float64},
)
    delta = nearest_periodic_delta_fractional(Float64.(a) .- Float64.(b), lattice)
    return norm(lattice * delta)
end

# -----------------------------------------------------------------------------
# Primitive-supercell relation
# -----------------------------------------------------------------------------

function factor_triples(n::Int)
    triples = NTuple{3,Int}[]
    for n1 in 1:n
        n % n1 == 0 || continue
        remaining = div(n, n1)
        for n2 in 1:remaining
            remaining % n2 == 0 || continue
            n3 = div(remaining, n2)
            push!(triples, (n1, n2, n3))
        end
    end
    return triples
end

function infer_repetitions(
    primitive_lattice::Matrix{Float64},
    supercell_lattice::Matrix{Float64},
    natoms_primitive::Int,
    natoms_supercell::Int,
)
    natoms_supercell % natoms_primitive == 0 ||
        error(
            "Supercell atom count ($natoms_supercell) is not divisible by " *
            "primitive-cell atom count ($natoms_primitive)."
        )

    Nc = div(natoms_supercell, natoms_primitive)

    # First verify that corresponding lattice vectors are approximately parallel.
    for i in 1:3
        a = primitive_lattice[:, i]
        A = supercell_lattice[:, i]
        cosangle = dot(a, A) / (norm(a) * norm(A))

        abs(abs(cosangle) - 1.0) <= LATTICE_DIRECTION_TOLERANCE ||
            error(
                "Primitive and supercell lattice vector $i are not sufficiently " *
                "parallel. This script currently assumes A_i = n_i a_i."
            )
    end

    length_ratios = ntuple(
        i -> norm(supercell_lattice[:, i]) / norm(primitive_lattice[:, i]),
        3,
    )

    candidates = factor_triples(Nc)
    isempty(candidates) && error("Could not factor number of primitive cells: $Nc.")

    best = first(candidates)
    best_score = Inf

    for candidate in candidates
        score = sum(
            (
                (candidate[i] - length_ratios[i]) /
                max(length_ratios[i], 1.0)
            )^2
            for i in 1:3
        )

        if score < best_score
            best = candidate
            best_score = score
        end
    end

    return best
end

# -----------------------------------------------------------------------------
# Raw finite-T average, primitive-site fitting, and symmetrization
# -----------------------------------------------------------------------------

function calculate_raw_average(
    xdatcar_file::AbstractString,
    start_snapshot::Int,
    final_snapshot::Union{Nothing,Int},
)
    anchor_fractional = nothing
    anchor_cartesian = nothing
    running_mean_cartesian = nothing
    expected_header = nothing

    count = 0
    snapshot_indices = Int[]
    marker_labels = Int[]

    selected_count, last_frame_seen = foreach_xdatcar_frame(
        xdatcar_file,
        start_snapshot,
        final_snapshot,
    ) do frame_index, marker_label, fractional_positions, header

        if isnothing(expected_header)
            expected_header = header
            anchor_fractional = copy(fractional_positions)
            anchor_cartesian = header.lattice * anchor_fractional
            running_mean_cartesian = copy(anchor_cartesian)
        else
            header.atom_species == expected_header.atom_species ||
                error("Atom species/order changed inside XDATCAR.")
            isapprox(
                header.lattice,
                expected_header.lattice;
                rtol=1.0e-10,
                atol=1.0e-10,
            ) || error(
                "The XDATCAR lattice changed. This script is for fixed-cell MD."
            )
        end

        if count == 0
            current_unwrapped_cartesian = copy(anchor_cartesian)
        else
            displacement_from_anchor = minimum_image_displacements(
                fractional_positions,
                anchor_fractional,
                header.lattice,
            )
            current_unwrapped_cartesian =
                anchor_cartesian + displacement_from_anchor
        end

        count += 1
        running_mean_cartesian .+=
            (current_unwrapped_cartesian .- running_mean_cartesian) ./ count

        push!(snapshot_indices, frame_index)
        push!(marker_labels, marker_label)

        if SHOW_PROGRESS && count % PROGRESS_EVERY == 0
            println(
                "Equilibrium pass: trajectory frame $frame_index " *
                "(marker $marker_label), selected count = $count"
            )
        end
    end

    count > 0 || error(
        "No trajectory frames were selected. Requested start = $start_snapshot, " *
        "final = $(isnothing(final_snapshot) ? "all" : string(final_snapshot)). " *
        "Last frame found in XDATCAR = $last_frame_seen."
    )

    if !isnothing(final_snapshot) && last(snapshot_indices) < final_snapshot
        error(
            "Requested --final $final_snapshot, but XDATCAR ended at frame " *
            "$(last(snapshot_indices))."
        )
    end

    return (
        header = expected_header,
        mean_cartesian = running_mean_cartesian,
        snapshot_indices = snapshot_indices,
        marker_labels = marker_labels,
    )
end

function assign_primitive_atoms(
    raw_mean_cartesian::Matrix{Float64},
    supercell_header::StructureHeader,
    primitive::Structure,
    equilibrium_primitive_lattice::Matrix{Float64},
    repetitions::NTuple{3,Int},
)
    natoms_sc = length(supercell_header.atom_species)
    natoms_prim = length(primitive.header.atom_species)
    Nc = prod(repetitions)

    primitive_coordinates =
        equilibrium_primitive_lattice \ raw_mean_cartesian

    atom_to_alpha = zeros(Int, natoms_sc)

    for atom in 1:natoms_sc
        species = supercell_header.atom_species[atom]
        candidates = findall(==(species), primitive.header.atom_species)

        isempty(candidates) &&
            error("Species $species exists in XDATCAR but not in POSCAR_primitive.")

        reduced = mod.(primitive_coordinates[:, atom], 1.0)

        distances = [
            periodic_distance(
                reduced,
                primitive.fractional_positions[:, alpha],
                equilibrium_primitive_lattice,
            )
            for alpha in candidates
        ]

        best_local = argmin(distances)
        best_alpha = candidates[best_local]
        best_distance = distances[best_local]

        best_distance <= POSITION_ASSIGNMENT_TOLERANCE_ANGSTROM ||
            error(
                "Could not assign supercell atom $atom ($species) to a primitive " *
                "atom label. Nearest relaxed primitive site is " *
                "$(round(best_distance, digits=6)) Å away. " *
                "This relaxed primitive structure may use a different origin, " *
                "orientation, or primitive-cell convention."
            )

        atom_to_alpha[atom] = best_alpha
    end

    counts = [count(==(alpha), atom_to_alpha) for alpha in 1:natoms_prim]
    for alpha in 1:natoms_prim
        counts[alpha] == Nc || error(
            "Primitive atom alpha=$alpha ($(primitive.header.atom_species[alpha])) " *
            "received $(counts[alpha]) supercell copies, but expected $Nc. " *
            "For structures with several same-species atoms per primitive cell, " *
            "the relaxed primitive sites may be too close for nearest-site " *
            "labelling; inspect the primitive-cell convention."
        )
    end

    return atom_to_alpha, primitive_coordinates
end

function fit_equilibrium_primitive_sites(
    primitive::Structure,
    primitive_coordinates::Matrix{Float64},
    atom_to_alpha::Vector{Int},
    equilibrium_primitive_lattice::Matrix{Float64},
)
    natoms_prim = length(primitive.header.atom_species)
    tau_T_unwrapped = zeros(Float64, 3, natoms_prim)
    counts = zeros(Int, natoms_prim)

    for atom in 1:length(atom_to_alpha)
        alpha = atom_to_alpha[atom]
        reduced = mod.(primitive_coordinates[:, atom], 1.0)

        # Unwrap this finite-T copy around the corresponding relaxed primitive
        # atom. The relaxed coordinate supplies only the alpha label/reference
        # branch; it is not imposed on the final average.
        delta = nearest_periodic_delta_fractional(
            reduced .- primitive.fractional_positions[:, alpha],
            equilibrium_primitive_lattice,
        )

        tau_T_unwrapped[:, alpha] .+=
            primitive.fractional_positions[:, alpha] .+ delta
        counts[alpha] += 1
    end

    for alpha in 1:natoms_prim
        counts[alpha] > 0 || error("No copies found for primitive atom $alpha.")
        tau_T_unwrapped[:, alpha] ./= counts[alpha]
    end

    return mod.(tau_T_unwrapped, 1.0)
end

function construct_symmetrized_equilibrium(
    raw_mean_cartesian::Matrix{Float64},
    supercell_header::StructureHeader,
    primitive::Structure;
    integer_cell_tolerance::Float64=INTEGER_CELL_TOLERANCE_ANGSTROM,
)
    natoms_sc = length(supercell_header.atom_species)
    natoms_prim = length(primitive.header.atom_species)

    repetitions = infer_repetitions(
        primitive.header.lattice,
        supercell_header.lattice,
        natoms_prim,
        natoms_sc,
    )

    Nc = prod(repetitions)

    equilibrium_primitive_lattice =
        supercell_header.lattice *
        Diagonal(1.0 ./ collect(repetitions))

    atom_to_alpha, primitive_coordinates = assign_primitive_atoms(
        raw_mean_cartesian,
        supercell_header,
        primitive,
        equilibrium_primitive_lattice,
        repetitions,
    )

    tau_T = fit_equilibrium_primitive_sites(
        primitive,
        primitive_coordinates,
        atom_to_alpha,
        equilibrium_primitive_lattice,
    )

    atom_to_cell = Vector{NTuple{3,Int}}(undef, natoms_sc)
    equilibrium_cartesian = zeros(Float64, 3, natoms_sc)
    residual_cartesian = zeros(Float64, 3, natoms_sc)

    seen = Set{Tuple{Int,Int,Int,Int}}()

    for atom in 1:natoms_sc
        alpha = atom_to_alpha[atom]

        L_raw, integer_residual = nearest_integer_translation(
            primitive_coordinates[:, atom] .- tau_T[:, alpha],
            equilibrium_primitive_lattice,
        )

        integer_residual <= integer_cell_tolerance ||
            error(
                "Raw averaged supercell atom $atom cannot be represented as " *
                "tau_alpha(T) + integer primitive translation within tolerance. " *
                "Residual = $(round(integer_residual, digits=6)) Å."
            )

        L = (
            mod(L_raw[1], repetitions[1]),
            mod(L_raw[2], repetitions[2]),
            mod(L_raw[3], repetitions[3]),
        )

        key = (alpha, L[1], L[2], L[3])
        key in seen && error(
            "Duplicate primitive mapping detected for alpha=$alpha, L=$L."
        )
        push!(seen, key)

        atom_to_cell[atom] = L

        equilibrium_cartesian[:, atom] .=
            equilibrium_primitive_lattice *
            (
                tau_T[:, alpha] .+
                collect(L)
            )

        residual_cartesian[:, atom] .=
            raw_mean_cartesian[:, atom] .-
            equilibrium_cartesian[:, atom]

        # Remove an arbitrary complete-supercell image from the residual.
        residual_fractional =
            supercell_header.lattice \ residual_cartesian[:, atom]
        residual_fractional =
            nearest_periodic_delta_fractional(
                residual_fractional,
                supercell_header.lattice,
            )
        residual_cartesian[:, atom] .=
            supercell_header.lattice * residual_fractional
    end

    length(seen) == natoms_sc ||
        error("Primitive mapping is not one-to-one.")

    equilibrium_fractional =
        supercell_header.lattice \ equilibrium_cartesian
    equilibrium_fractional .= mod.(equilibrium_fractional, 1.0)

    equilibrium_supercell =
        Structure(supercell_header, equilibrium_fractional)

    primitive_header_T = StructureHeader(
        "Finite-temperature primitive equilibrium",
        equilibrium_primitive_lattice,
        primitive.header.species_names,
        primitive.header.species_counts,
        primitive.header.atom_species,
    )
    equilibrium_primitive =
        Structure(primitive_header_T, tau_T)

    mapping = AtomPrimitiveMapping(
        repetitions,
        Nc,
        natoms_prim,
        natoms_sc,
        atom_to_alpha,
        atom_to_cell,
        tau_T,
        equilibrium_primitive_lattice,
    )

    return (
        equilibrium_supercell = equilibrium_supercell,
        equilibrium_primitive = equilibrium_primitive,
        mapping = mapping,
        symmetry_residual_cartesian = residual_cartesian,
    )
end

# -----------------------------------------------------------------------------
# POSCAR writing
# -----------------------------------------------------------------------------

function write_poscar(
    filename::AbstractString,
    structure::Structure;
    comment::AbstractString = structure.header.comment,
)
    open(filename, "w") do io
        println(io, comment)
        println(io, "1.0")

        lattice = structure.header.lattice
        for i in 1:3
            @printf(
                io,
                "  %20.14f  %20.14f  %20.14f\n",
                lattice[1, i],
                lattice[2, i],
                lattice[3, i],
            )
        end

        println(io, join(structure.header.species_names, "  "))
        println(io, join(structure.header.species_counts, "  "))
        println(io, "Direct")

        for atom in 1:size(structure.fractional_positions, 2)
            @printf(
                io,
                "  %20.14f  %20.14f  %20.14f\n",
                structure.fractional_positions[1, atom],
                structure.fractional_positions[2, atom],
                structure.fractional_positions[3, atom],
            )
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Covariance pass
# -----------------------------------------------------------------------------

function calculate_covariance(
    xdatcar_file::AbstractString,
    start_snapshot::Int,
    final_snapshot::Union{Nothing,Int},
    equilibrium::Structure,
    expected_snapshot_indices::Vector{Int},
    expected_marker_labels::Vector{Int},
)
    natoms = length(equilibrium.header.atom_species)
    ndof = 3 * natoms

    count = 0
    running_mean = zeros(Float64, ndof)
    M2 = zeros(Float64, ndof, ndof)

    snapshot_indices = Int[]
    marker_labels = Int[]

    foreach_xdatcar_frame(
        xdatcar_file,
        start_snapshot,
        final_snapshot,
    ) do frame_index, marker_label, fractional_positions, header

        header.atom_species == equilibrium.header.atom_species ||
            error("Atom species/order changed between covariance and equilibrium.")

        isapprox(
            header.lattice,
            equilibrium.header.lattice;
            rtol=1.0e-10,
            atol=1.0e-10,
        ) || error("XDATCAR lattice differs from equilibrium lattice.")

        displacements = minimum_image_displacements(
            fractional_positions,
            equilibrium.fractional_positions,
            equilibrium.header.lattice,
        )
        u = vec(displacements)

        count += 1
        delta = u .- running_mean
        running_mean .+= delta ./ count
        delta_after = u .- running_mean

        # M2 += delta * delta_after'
        LinearAlgebra.BLAS.ger!(1.0, delta, delta_after, M2)

        push!(snapshot_indices, frame_index)
        push!(marker_labels, marker_label)

        if SHOW_PROGRESS && count % PROGRESS_EVERY == 0
            println(
                "Covariance pass: trajectory frame $frame_index " *
                "(marker $marker_label), selected count = $count"
            )
        end
    end

    snapshot_indices == expected_snapshot_indices ||
        error("The covariance pass read a different snapshot set than the mean pass.")
    marker_labels == expected_marker_labels ||
        error("XDATCAR marker labels changed between passes.")

    count > 0 || error("No frames were used in covariance pass.")

    covariance = M2 ./ count   # canonical/population normalization 1/N
    covariance .= (covariance + covariance') ./ 2

    return running_mean, covariance
end

# -----------------------------------------------------------------------------
# HDF5 output
# -----------------------------------------------------------------------------

function write_covariance_file(
    filename::AbstractString,
    settings,
    raw_average,
    equilibrium_result,
    primitive_relaxed::Structure,
    mean_u::Vector{Float64},
    covariance::Matrix{Float64},
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    equilibrium = equilibrium_result.equilibrium_supercell
    equilibrium_primitive = equilibrium_result.equilibrium_primitive
    mapping = equilibrium_result.mapping
    residual = equilibrium_result.symmetry_residual_cartesian

    natoms = mapping.number_of_supercell_atoms
    ndof = 3 * natoms

    dof_supercell_atom = repeat(collect(1:natoms); inner=3)
    dof_component_index = repeat(collect(1:3), natoms)
    dof_component = repeat(["x", "y", "z"], natoms)

    dof_alpha = [
        mapping.atom_to_primitive_atom[atom]
        for atom in dof_supercell_atom
    ]

    dof_L = hcat([
        collect(mapping.atom_to_cell[atom])
        for atom in dof_supercell_atom
    ]...)

    raw_mean_fractional =
        equilibrium.header.lattice \ raw_average.mean_cartesian
    raw_mean_fractional .= mod.(raw_mean_fractional, 1.0)

    thermal_shift_fractional = zeros(
        Float64,
        3,
        mapping.number_of_primitive_atoms,
    )

    for alpha in 1:mapping.number_of_primitive_atoms
        thermal_shift_fractional[:, alpha] .=
            nearest_periodic_delta_fractional(
                equilibrium_primitive.fractional_positions[:, alpha] .-
                primitive_relaxed.fractional_positions[:, alpha],
                equilibrium_primitive.header.lattice,
            )
    end

    h5open(temporary, "w") do f
        f["format_version"] = 2
        f["primitive_projection_tolerance_angstrom"] = settings[:projection_tolerance]
        f["number_of_snapshots"] = length(raw_average.snapshot_indices)
        f["snapshot_indices"] = raw_average.snapshot_indices
        f["xdatcar_marker_labels"] = raw_average.marker_labels
        native=is_mace_file(settings[:xdatcar])
        f["trajectory_format"]=native ? "LAMMPS custom dump (MACE)" : "VASP XDATCAR"
        if native
            f["lammps_timesteps"]=[MACE_TIMESTEPS[i] for i in raw_average.snapshot_indices]
            f["native_atom_ids"]=MaceTrajectory.first_frame(settings[:xdatcar]).ids
        end
        f["requested_start_snapshot"] = settings[:start]
        f["requested_final_snapshot"] =
            isnothing(settings[:final]) ? -1 : settings[:final]
        f["actual_first_snapshot"] = first(raw_average.snapshot_indices)
        f["actual_last_snapshot"] = last(raw_average.snapshot_indices)
        f["number_of_atoms"] = natoms
        f["number_of_degrees_of_freedom"] = ndof

        sources = create_group(f, "sources")
        sources["trajectory_file"] = abspath(settings[:xdatcar])
        sources["xdatcar_file"] = native ? joinpath(NOBACKUP_DIR,"SC_XDATCAR") : abspath(settings[:xdatcar])
        sources["relaxed_primitive_poscar"] = abspath(settings[:primitive])
        sources["equilibrium_supercell_poscar"] =
            abspath(settings[:equilibrium_output])
        sources["equilibrium_primitive_poscar"] =
            abspath(settings[:primitive_equilibrium_output])

        geometry = create_group(f, "geometry")
        geometry["supercell_lattice_angstrom"] =
            equilibrium.header.lattice
        geometry["raw_mean_fractional_positions"] =
            raw_mean_fractional
        geometry["raw_mean_cartesian_unwrapped_angstrom"] =
            raw_average.mean_cartesian
        geometry["equilibrium_fractional_positions"] =
            equilibrium.fractional_positions
        geometry["equilibrium_primitive_lattice_angstrom"] =
            equilibrium_primitive.header.lattice
        geometry["equilibrium_primitive_fractional_positions"] =
            equilibrium_primitive.fractional_positions
        geometry["relaxed_primitive_fractional_positions"] =
            primitive_relaxed.fractional_positions
        geometry["primitive_thermal_shift_fractional"] =
            thermal_shift_fractional
        geometry["symmetry_projection_residual_angstrom"] =
            residual
        geometry["atom_species"] =
            equilibrium.header.atom_species

        mapping_group = create_group(f, "mapping")
        mapping_group["supercell_atom"] = collect(1:natoms)
        mapping_group["primitive_atom_alpha"] =
            mapping.atom_to_primitive_atom
        mapping_group["cell_L"] =
            hcat([collect(L) for L in mapping.atom_to_cell]...)
        mapping_group["repetitions_n1_n2_n3"] =
            collect(mapping.repetitions)
        mapping_group["number_of_primitive_cells"] =
            mapping.number_of_primitive_cells
        mapping_group["number_of_primitive_atoms"] =
            mapping.number_of_primitive_atoms

        dof = create_group(f, "dof")
        dof["supercell_atom"] = dof_supercell_atom
        dof["primitive_atom_alpha"] = dof_alpha
        dof["cell_L"] = dof_L
        dof["component_index"] = dof_component_index
        dof["component"] = dof_component

        covariance_group = create_group(f, "covariance")
        covariance_group["mean_u_angstrom"] = mean_u
        covariance_group["covariance_angstrom2"] = covariance
        covariance_group["normalization"] = "1/N"
    end

    mv(temporary, filename; force=true)
    return nothing
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function link_preparation_outputs(settings)
    for path in (joinpath(NOBACKUP_DIR,"SC_POSCAR"),settings[:primitive],
                 settings[:equilibrium_output],settings[:primitive_equilibrium_output],
                 settings[:covariance_output],joinpath(NOBACKUP_DIR,"SC_XDATCAR"))
        isfile(path) || continue
        destination=joinpath(CALC_DIR,basename(path))
        abspath(destination)==abspath(path) && continue
        if islink(destination)
            realpath(destination)==realpath(path) && continue
            rm(destination)
        elseif ispath(destination)
            error("Cannot replace input with preparation link: $destination")
        end
        symlink(path,destination)
    end
end

function prepare_mace_inputs(settings)
    frame=MaceTrajectory.first_frame(settings[:xdatcar])
    frame === nothing && error("Empty MACE trajectory")
    nc=prod(settings[:repetitions])
    all(count -> count % nc == 0, frame.counts) || error("Species counts are not multiples of the number of primitive cells")
    structure=Structure(StructureHeader("First MACE frame",frame.lattice,frame.species,
                        frame.counts,frame.atom_species), frame.fractional)
    sc=joinpath(NOBACKUP_DIR,"SC_POSCAR")
    !settings[:overwrite] && isfile(sc) && error("Output exists: $sc")
    write_poscar(sc,structure;comment="First MACE frame (not thermal equilibrium)")
    if !isfile(settings[:primitive])
        MaterialPaths.check_storage_path(settings[:primitive])
        sites=MaceTrajectory.primitive_sites(frame,settings[:repetitions])
        primitive=Structure(StructureHeader("Primitive topology from MACE frame 1",sites.lattice,
                            sites.species,sites.counts,sites.atom_species),sites.fractional)
        write_poscar(settings[:primitive],primitive)
    end
    # HAMSTER's existing chunk writer needs a VASP trajectory. Covariance itself
    # always reads the native MACE source, retaining the original estimator.
    if settings[:reuse_xdatcar]
        isfile(joinpath(NOBACKUP_DIR,"SC_XDATCAR")) || error("No completed SC_XDATCAR to reuse")
        progress_println("Reusing completed SC_XDATCAR (native MACE remains the covariance input).")
    else
        MaceTrajectory.export_xdatcar(settings[:xdatcar],joinpath(NOBACKUP_DIR,"SC_XDATCAR");
                                      overwrite=settings[:overwrite])
    end
    progress_println("Prepared initial POSCARs and full HAMSTER SC_XDATCAR under ",NOBACKUP_DIR)
end

function main()
    settings = parse_arguments(ARGS)
    isfile(settings[:xdatcar]) || error("Missing trajectory: $(settings[:xdatcar])")
    for key in (:equilibrium_output,:primitive_equilibrium_output,:covariance_output)
        MaterialPaths.check_storage_path(settings[key])
        !settings[:overwrite] && ispath(settings[key]) && error("Output exists: $(settings[key])")
    end
    is_mace_file(settings[:xdatcar]) && prepare_mace_inputs(settings)

    progress_println("Covariance BLAS threads: ", BLAS.get_num_threads())
    progress_println("Trajectory: ", settings[:xdatcar])
    progress_println("Primitive reference: ", settings[:primitive])
    progress_println("Start snapshot: ", settings[:start])
    SHOW_PROGRESS && println(
        "Final snapshot: ",
        isnothing(settings[:final]) ? "all" : settings[:final],
    )

    for path in (
        settings[:xdatcar],
        settings[:primitive],
    )
        isfile(path) || error("Required input file does not exist: $path")
    end

    if !settings[:overwrite]
        for path in (
            settings[:equilibrium_output],
            settings[:primitive_equilibrium_output],
            settings[:covariance_output],
        )
            isfile(path) && error("Output exists: $path")
        end
    end

    primitive = read_poscar(settings[:primitive])

    # -------------------------------------------------------------------------
    # Pass 1: raw finite-T trajectory average
    # -------------------------------------------------------------------------

    raw_average = calculate_raw_average(
        settings[:xdatcar],
        settings[:start],
        settings[:final],
    )

    SHOW_PROGRESS && println(
        "Selected $(length(raw_average.snapshot_indices)) trajectory frame(s): " *
        "$(first(raw_average.snapshot_indices)):" *
        "$(last(raw_average.snapshot_indices))"
    )

    # -------------------------------------------------------------------------
    # Fit finite-T primitive sites and project the mean structure exactly onto
    # primitive translational symmetry.
    # -------------------------------------------------------------------------

    equilibrium_result = construct_symmetrized_equilibrium(
        raw_average.mean_cartesian,
        raw_average.header,
        primitive;
        integer_cell_tolerance=settings[:projection_tolerance],
    )

    mapping = equilibrium_result.mapping

    SHOW_PROGRESS && println(
        "Inferred supercell repetitions: ",
        mapping.repetitions,
        "  (Nc = ",
        mapping.number_of_primitive_cells,
        ")",
    )

    max_symmetry_residual =
        maximum(norm(equilibrium_result.symmetry_residual_cartesian[:, atom])
                for atom in 1:mapping.number_of_supercell_atoms)

    SHOW_PROGRESS && println(
        "Maximum raw-average -> primitive-symmetry residual: ",
        max_symmetry_residual,
        " Å",
    )

    write_poscar(
        settings[:equilibrium_output],
        equilibrium_result.equilibrium_supercell;
        comment =
            "Primitive-periodic finite-T equilibrium from the MD trajectory frames " *
            "$(first(raw_average.snapshot_indices))-$(last(raw_average.snapshot_indices))",
    )

    write_poscar(
        settings[:primitive_equilibrium_output],
        equilibrium_result.equilibrium_primitive;
        comment =
            "Finite-T primitive equilibrium extracted from symmetrized XDATCAR average",
    )

    progress_println("Wrote: ", settings[:equilibrium_output])
    progress_println("Wrote: ", settings[:primitive_equilibrium_output])

    # -------------------------------------------------------------------------
    # Pass 2: displacement covariance around the final symmetrized equilibrium.
    # -------------------------------------------------------------------------

    mean_u, covariance = calculate_covariance(
        settings[:xdatcar],
        settings[:start],
        settings[:final],
        equilibrium_result.equilibrium_supercell,
        raw_average.snapshot_indices,
        raw_average.marker_labels,
    )

    write_covariance_file(
        settings[:covariance_output],
        settings,
        raw_average,
        equilibrium_result,
        primitive,
        mean_u,
        covariance,
    )

    progress_println("Wrote: ", settings[:covariance_output])
    SHOW_PROGRESS && println(
        "Maximum |mean u| after primitive-symmetry centering = ",
        maximum(abs.(mean_u)),
        " Å",
    )
    progress_println("Covariance dimension = ", size(covariance))
    link_preparation_outputs(settings)
    progress_println("Done.")

    return nothing
end

main()
