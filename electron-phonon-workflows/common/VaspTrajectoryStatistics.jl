module VaspTrajectoryStatistics

using LinearAlgebra
using Printf

export VaspHeader,
       VaspStructure,
       read_poscar,
       write_poscar,
       foreach_selected_xdatcar_frame,
       minimum_image_displacements,
       enumerate_cells,
       canonical_cell,
       component_index,
       component_label

const CellVector = NTuple{3, Int}

struct VaspHeader
    comment::String
    lattice::Matrix{Float64}       # 3×3, columns are lattice vectors, Å
    coordinate_scale::Vector{Float64}
    species::Vector{String}
    counts::Vector{Int}
    atom_species::Vector{String}
end

struct VaspStructure
    header::VaspHeader
    fractional_positions::Matrix{Float64}  # 3×Natoms
end

function parse_float_triplet(line::AbstractString)
    fields = split(strip(line))
    length(fields) >= 3 || error("Expected three numbers, found: $line")
    return parse.(Float64, fields[1:3])
end

function make_scaled_lattice(
    scale_tokens::Vector{Float64},
    raw_rows::Matrix{Float64},
)
    if length(scale_tokens) == 1
        input_scale = only(scale_tokens)
        input_scale == 0 && error("A VASP scaling factor cannot be zero.")

        scale = if input_scale > 0
            input_scale
        else
            target_volume = -input_scale
            raw_volume = abs(det(raw_rows))
            raw_volume > 0 || error("The unscaled lattice has zero volume.")
            (target_volume / raw_volume)^(1 / 3)
        end

        scaled_rows = scale .* raw_rows
        return permutedims(scaled_rows), fill(scale, 3)
    elseif length(scale_tokens) == 3
        all(scale_tokens .> 0) ||
            error("Three VASP scaling factors must all be positive.")
        scaled_rows = raw_rows .* reshape(scale_tokens, 1, 3)
        return permutedims(scaled_rows), copy(scale_tokens)
    end

    error("The VASP scale line must contain one or three numbers.")
end

function expand_atom_species(species::Vector{String}, counts::Vector{Int})
    length(species) == length(counts) ||
        error("Species and count lists have different lengths.")

    result = String[]
    for (name, count) in zip(species, counts)
        append!(result, fill(name, count))
    end
    return result
end

function read_vasp_header(io)::VaspHeader
    eof(io) && error("Unexpected end of file while reading a VASP header.")
    comment = chomp(readline(io))
    scale_tokens = parse.(Float64, split(strip(readline(io))))

    raw_rows = zeros(Float64, 3, 3)
    for row in 1:3
        raw_rows[row, :] .= parse_float_triplet(readline(io))
    end
    lattice, coordinate_scale = make_scaled_lattice(scale_tokens, raw_rows)

    species_or_counts = split(strip(readline(io)))
    old_format = all(token -> tryparse(Int, token) !== nothing, species_or_counts)
    old_format && error(
        "Old-style POSCAR/XDATCAR without explicit species names is not " *
        "supported because atom labels are required."
    )

    species = String.(species_or_counts)
    counts = parse.(Int, split(strip(readline(io))))
    atom_species = expand_atom_species(species, counts)

    return VaspHeader(
        comment,
        lattice,
        coordinate_scale,
        species,
        counts,
        atom_species,
    )
end

function read_poscar(filename::AbstractString)::VaspStructure
    isfile(filename) || error("POSCAR does not exist: $filename")

    return open(filename, "r") do io
        header = read_vasp_header(io)
        natoms = sum(header.counts)

        mode_line = strip(readline(io))
        if startswith(lowercase(mode_line), "s")
            mode_line = strip(readline(io))
        end

        cartesian_mode = startswith(lowercase(mode_line), "c") ||
                         startswith(lowercase(mode_line), "k")

        coordinates = zeros(Float64, 3, natoms)
        for atom in 1:natoms
            coordinates[:, atom] .= parse_float_triplet(readline(io))
        end

        fractional = if cartesian_mode
            cartesian = header.coordinate_scale .* coordinates
            header.lattice \ cartesian
        else
            coordinates
        end

        return VaspStructure(header, fractional)
    end
end

function write_poscar(
    filename::AbstractString,
    structure::VaspStructure;
    comment::AbstractString="Trajectory-averaged equilibrium structure",
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    open(temporary, "w") do io
        println(io, comment)
        println(io, "1.0")

        lattice_rows = permutedims(structure.header.lattice)
        for row in 1:3
            @printf(io, " %20.14f %20.14f %20.14f\n",
                lattice_rows[row, 1],
                lattice_rows[row, 2],
                lattice_rows[row, 3],
            )
        end

        println(io, join(structure.header.species, " "))
        println(io, join(structure.header.counts, " "))
        println(io, "Direct")

        positions = mod.(structure.fractional_positions, 1.0)
        for atom in axes(positions, 2)
            @printf(io, " %20.14f %20.14f %20.14f\n",
                positions[1, atom],
                positions[2, atom],
                positions[3, atom],
            )
        end
    end

    mv(temporary, filename; force=true)
    return nothing
end

"""
Iterate through selected XDATCAR frames without loading the complete trajectory.

The frame index is the one-based ordinal position in XDATCAR, matching the
configuration indices used by HAMSTER when it selects trajectory frames.
"""
function foreach_selected_xdatcar_frame(
    callback::Function,
    filename::AbstractString,
    wanted_indices::Vector{Int},
)
    isfile(filename) || error("XDATCAR does not exist: $filename")
    isempty(wanted_indices) && return nothing
    all(wanted_indices .>= 1) || error("XDATCAR frame indices must be positive.")

    wanted = Set(wanted_indices)
    maximum_wanted = maximum(wanted_indices)
    found = Set{Int}()

    open(filename, "r") do io
        header = read_vasp_header(io)
        natoms = sum(header.counts)
        frame_index = 0

        while !eof(io) && frame_index < maximum_wanted
            marker_line = strip(readline(io))
            isempty(marker_line) && continue

            marker_lower = lowercase(marker_line)
            is_direct = startswith(marker_lower, "direct")
            is_cartesian = startswith(marker_lower, "cartesian") ||
                           startswith(marker_lower, "kpoints")

            ((is_direct || is_cartesian) && occursin("configuration", marker_lower)) ||
                error(
                    "Expected an XDATCAR configuration marker, found: " *
                    "\"$marker_line\"."
                )

            frame_index += 1
            marker_match = match(r"configuration\s*=\s*([0-9]+)", marker_lower)
            marker_label = isnothing(marker_match) ?
                           frame_index : parse(Int, marker_match.captures[1])

            coordinates = zeros(Float64, 3, natoms)
            for atom in 1:natoms
                eof(io) && error("XDATCAR ended inside frame $frame_index.")
                coordinates[:, atom] .= parse_float_triplet(readline(io))
            end

            if frame_index in wanted
                fractional = if is_cartesian
                    cartesian = header.coordinate_scale .* coordinates
                    header.lattice \ cartesian
                else
                    coordinates
                end

                callback(frame_index, marker_label, fractional, header)
                push!(found, frame_index)
            end
        end
    end

    missing = sort(collect(setdiff(wanted, found)))
    isempty(missing) ||
        error("XDATCAR does not contain requested frame(s): $(join(missing, ", ")).")

    return nothing
end

const IMAGE_OFFSETS = [
    (i, j, k) for i in -2:2 for j in -2:2 for k in -2:2
]

"""
Return Cartesian displacements `current - reference`, choosing the shortest
periodic image in a possibly non-orthogonal cell.
"""
function minimum_image_displacements(
    current_fractional::Matrix{Float64},
    reference_fractional::Matrix{Float64},
    lattice::Matrix{Float64},
)
    size(current_fractional) == size(reference_fractional) ||
        error("Current and reference structures have different atom counts.")

    natoms = size(current_fractional, 2)
    displacements = zeros(Float64, 3, natoms)

    for atom in 1:natoms
        delta_fractional = current_fractional[:, atom] -
                           reference_fractional[:, atom]
        nearest_integer = round.(Int, delta_fractional)

        best_norm2 = Inf
        best_cartesian = zeros(Float64, 3)

        for offset in IMAGE_OFFSETS
            image_shift = nearest_integer .+ collect(offset)
            candidate_fractional = delta_fractional .- image_shift
            candidate_cartesian = lattice * candidate_fractional
            candidate_norm2 = dot(candidate_cartesian, candidate_cartesian)

            if candidate_norm2 < best_norm2
                best_norm2 = candidate_norm2
                best_cartesian .= candidate_cartesian
            end
        end

        displacements[:, atom] .= best_cartesian
    end

    return displacements
end

function enumerate_cells(repetitions::CellVector)
    n1, n2, n3 = repetitions
    return [
        (r1, r2, r3)
        for r3 in 0:(n3 - 1)
        for r2 in 0:(n2 - 1)
        for r1 in 0:(n1 - 1)
    ]
end

function canonical_cell(cell::CellVector, repetitions::CellVector)::CellVector
    return (
        mod(cell[1], repetitions[1]),
        mod(cell[2], repetitions[2]),
        mod(cell[3], repetitions[3]),
    )
end

function canonical_cell(
    a::CellVector,
    b::CellVector,
    repetitions::CellVector,
)::CellVector
    return canonical_cell(
        (a[1] + b[1], a[2] + b[2], a[3] + b[3]),
        repetitions,
    )
end

function component_index(label::AbstractString)
    lower = lowercase(String(label))
    lower == "x" && return 1
    lower == "y" && return 2
    lower == "z" && return 3
    error("Unknown Cartesian component label: $label")
end

function component_label(index::Integer)
    index == 1 && return "x"
    index == 2 && return "y"
    index == 3 && return "z"
    error("Cartesian component index must be 1, 2, or 3.")
end

end # module VaspTrajectoryStatistics
