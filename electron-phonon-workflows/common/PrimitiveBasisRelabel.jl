module PrimitiveBasisRelabel

using LinearAlgebra

export PrimitiveOrbitalAddress,
       PrimitiveBasisMap,
       build_primitive_basis_map,
       relabel_basis_index,
       relabel_matrix_element,
       primitive_basis_mapping_table

const CellVector = NTuple{3, Int}

struct SimplePoscar
    lattice::Matrix{Float64}                 # columns are lattice vectors, Å
    species::Vector{String}
    counts::Vector{Int}
    atom_species::Vector{String}
    fractional_positions::Matrix{Float64}    # 3 × Natom
end

struct PrimitiveOrbitalAddress
    basis_index::Int
    supercell_atom::Int
    primitive_orbital::Int                   # μ inside the primitive cell
    primitive_atom::Int
    local_orbital::Int
    orbital_token::String
    basis_label::String
    spin_slot::Int
    cell::CellVector                         # primitive-cell vector R
end

struct PrimitiveBasisMap
    repetitions::CellVector                  # (n1,n2,n3)
    supercell_matrix::Matrix{Int}
    number_of_primitive_cells::Int
    number_of_primitive_atoms::Int
    number_of_supercell_atoms::Int
    spin_factor::Int
    atom_to_primitive_atom::Vector{Int}
    atom_to_cell::Vector{CellVector}
    index_to_address::Vector{PrimitiveOrbitalAddress}
end

# -----------------------------------------------------------------------------
# POSCAR reader
# -----------------------------------------------------------------------------

function parse_triplet(line::AbstractString)
    fields = split(strip(line))
    length(fields) >= 3 || error("Expected three numbers, found: $line")
    return parse.(Float64, fields[1:3])
end

function scaled_lattice(scale_tokens::Vector{Float64}, raw_rows::Matrix{Float64})
    if length(scale_tokens) == 1
        input_scale = only(scale_tokens)
        input_scale == 0 && error("POSCAR scaling factor cannot be zero.")
        scale = if input_scale > 0
            input_scale
        else
            raw_volume = abs(det(raw_rows))
            raw_volume > 0 || error("POSCAR lattice has zero volume.")
            (-input_scale / raw_volume)^(1 / 3)
        end
        return permutedims(scale .* raw_rows), fill(scale, 3)
    elseif length(scale_tokens) == 3
        all(scale_tokens .> 0) ||
            error("Three POSCAR scale factors must all be positive.")
        scaled_rows = raw_rows .* reshape(scale_tokens, 1, 3)
        return permutedims(scaled_rows), copy(scale_tokens)
    end
    error("POSCAR scale line must contain one or three numbers.")
end

function expand_species(species::Vector{String}, counts::Vector{Int})
    length(species) == length(counts) ||
        error("POSCAR species and count lists have different lengths.")
    result = String[]
    for (name, count) in zip(species, counts)
        append!(result, fill(name, count))
    end
    return result
end

function read_simple_poscar(filename::AbstractString)::SimplePoscar
    isfile(filename) || error("POSCAR file does not exist: $filename")

    return open(filename, "r") do io
        eof(io) && error("Empty POSCAR: $filename")
        readline(io)  # comment
        scale_tokens = parse.(Float64, split(strip(readline(io))))

        raw_rows = zeros(Float64, 3, 3)
        for row in 1:3
            raw_rows[row, :] .= parse_triplet(readline(io))
        end
        lattice, coordinate_scale = scaled_lattice(scale_tokens, raw_rows)

        species_or_counts = split(strip(readline(io)))
        old_vasp_format = all(x -> tryparse(Int, x) !== nothing, species_or_counts)
        if old_vasp_format
            error(
                "POSCAR $filename uses the old format without element names. " *
                "Explicit species labels are required for primitive-cell mapping."
            )
        end

        species = String.(species_or_counts)
        counts = parse.(Int, split(strip(readline(io))))
        atom_species = expand_species(species, counts)
        natoms = sum(counts)

        coordinate_line = strip(readline(io))
        if startswith(lowercase(coordinate_line), "s")
            coordinate_line = strip(readline(io))
        end

        cartesian = startswith(lowercase(coordinate_line), "c") ||
                    startswith(lowercase(coordinate_line), "k")

        positions = zeros(Float64, 3, natoms)
        for atom in 1:natoms
            positions[:, atom] .= parse_triplet(readline(io))
        end

        fractional = if cartesian
            cartesian_positions = coordinate_scale .* positions
            lattice \ cartesian_positions
        else
            positions
        end

        return SimplePoscar(
            lattice,
            species,
            counts,
            atom_species,
            fractional,
        )
    end
end

# -----------------------------------------------------------------------------
# hconf orbital parsing
# -----------------------------------------------------------------------------

function remove_comment(line::AbstractString)
    return strip(first(split(String(line), '#'; limit=2)))
end

"""
Read the orbital token list from every `begin atom ELEMENT ... end` block.
Each token on the `orbitals = ...` line is one HAMSTER orbital attached to
that atom, matching the ion-major/orbital-major ordering used by HAMSTER.
"""
function read_hconf_orbitals(filename::AbstractString)
    isfile(filename) || error("HAMSTER configuration file does not exist: $filename")

    result = Dict{String, Vector{String}}()
    current_species = nothing

    for raw_line in eachline(filename)
        line = remove_comment(raw_line)
        isempty(line) && continue

        block_match = match(r"(?i)^begin\s+atom\s+(\S+)\s*$", line)
        if !isnothing(block_match)
            current_species = block_match.captures[1]
            haskey(result, current_species) &&
                error("Duplicate atom block for $current_species in $filename.")
            result[current_species] = String[]
            continue
        end

        if lowercase(line) == "end"
            current_species = nothing
            continue
        end

        isnothing(current_species) && continue

        orbital_match = match(r"(?i)^orbitals\s*=\s*(.*)$", line)
        if !isnothing(orbital_match)
            tokens = String.(split(strip(orbital_match.captures[1])))
            isempty(tokens) && error("Empty orbitals list for $current_species.")
            append!(result[current_species], tokens)
        end
    end

    isempty(result) && error("No `begin atom` blocks were found in $filename.")
    for (species, orbitals) in result
        isempty(orbitals) && error("No orbitals were found for atom type $species.")
    end
    return result
end

# -----------------------------------------------------------------------------
# Lattice and atom mapping
# -----------------------------------------------------------------------------

function infer_diagonal_repetitions(
    primitive_lattice::Matrix{Float64},
    supercell_lattice::Matrix{Float64};
    tolerance::Float64=1.0e-7,
)
    raw = primitive_lattice \ supercell_lattice
    integer_matrix = round.(Int, raw)

    isapprox(raw, integer_matrix; rtol=0.0, atol=tolerance) ||
        error(
            "The supercell lattice is not an integer combination of the " *
            "primitive lattice. A_primitive\\A_supercell =\n$raw"
        )

    off_diagonal = integer_matrix - Diagonal(diag(integer_matrix))
    all(off_diagonal .== 0) ||
        error(
            "The inferred integer supercell matrix is not diagonal:\n" *
            "$integer_matrix\nThis implementation currently expects " *
            "A_sc = A_prim * diag(n1,n2,n3)."
        )

    repetitions = Tuple(diag(integer_matrix))
    all(x -> x > 0, repetitions) ||
        error("Inferred repetitions must be positive; found $repetitions.")

    reconstructed = primitive_lattice * Float64.(integer_matrix)
    isapprox(reconstructed, supercell_lattice; rtol=tolerance, atol=tolerance) ||
        error("Failed to reconstruct the supercell lattice from the primitive cell.")

    return repetitions, integer_matrix
end

centered_fractional(x::AbstractVector) = x .- round.(x)

function canonical_cell(raw_cell::AbstractVector{<:Integer}, repetitions::CellVector)
    return (
        mod(raw_cell[1], repetitions[1]),
        mod(raw_cell[2], repetitions[2]),
        mod(raw_cell[3], repetitions[3]),
    )
end

function try_atom_mapping(
    primitive::SimplePoscar,
    supercell::SimplePoscar,
    repetitions::CellVector,
    origin_shift::Vector{Float64};
    position_tolerance_angstrom::Float64,
)
    nsuper = length(supercell.atom_species)
    atom_to_primitive = zeros(Int, nsuper)
    atom_to_cell = Vector{CellVector}(undef, nsuper)
    max_residual = 0.0

    # Express every supercell atom in primitive fractional coordinates.
    super_cartesian = supercell.lattice * supercell.fractional_positions
    super_in_primitive = primitive.lattice \ super_cartesian

    for super_atom in 1:nsuper
        species = supercell.atom_species[super_atom]
        candidates = findall(==(species), primitive.atom_species)
        isempty(candidates) &&
            error("Species $species is absent from POSCAR_primitive.")

        best_primitive = 0
        best_raw_cell = zeros(Int, 3)
        best_residual = Inf

        for primitive_atom in candidates
            delta = super_in_primitive[:, super_atom] .-
                    origin_shift .-
                    primitive.fractional_positions[:, primitive_atom]
            raw_cell = round.(Int, delta)
            residual_cartesian = primitive.lattice * (delta .- raw_cell)
            residual = norm(residual_cartesian)

            if residual < best_residual
                best_residual = residual
                best_primitive = primitive_atom
                best_raw_cell .= raw_cell
            end
        end

        best_residual <= position_tolerance_angstrom || return nothing
        atom_to_primitive[super_atom] = best_primitive
        atom_to_cell[super_atom] = canonical_cell(best_raw_cell, repetitions)
        max_residual = max(max_residual, best_residual)
    end

    pairs = collect(zip(atom_to_primitive, atom_to_cell))
    length(unique(pairs)) == length(pairs) || return nothing

    expected_count = length(primitive.atom_species) * prod(repetitions)
    length(pairs) == expected_count || return nothing

    return (
        atom_to_primitive=atom_to_primitive,
        atom_to_cell=atom_to_cell,
        max_residual=max_residual,
        origin_shift=origin_shift,
    )
end

function map_supercell_atoms(
    primitive::SimplePoscar,
    supercell::SimplePoscar,
    repetitions::CellVector;
    position_tolerance_angstrom::Float64=1.0e-4,
)
    expected_atoms = length(primitive.atom_species) * prod(repetitions)
    length(supercell.atom_species) == expected_atoms ||
        error(
            "The supercell contains $(length(supercell.atom_species)) atoms, " *
            "but POSCAR_primitive and repetitions $repetitions imply " *
            "$expected_atoms atoms."
        )

    super_cartesian = supercell.lattice * supercell.fractional_positions
    super_in_primitive = primitive.lattice \ super_cartesian

    first_species = supercell.atom_species[1]
    first_candidates = findall(==(first_species), primitive.atom_species)
    isempty(first_candidates) && error("Species $first_species is absent from primitive POSCAR.")

    successful = NamedTuple[]
    for primitive_atom in first_candidates
        raw_shift = super_in_primitive[:, 1] .-
                    primitive.fractional_positions[:, primitive_atom]
        origin_shift = collect(centered_fractional(raw_shift))
        candidate = try_atom_mapping(
            primitive,
            supercell,
            repetitions,
            origin_shift;
            position_tolerance_angstrom=position_tolerance_angstrom,
        )
        isnothing(candidate) || push!(successful, candidate)
    end

    isempty(successful) &&
        error(
            "Could not map the equilibrium supercell atoms onto primitive-cell " *
            "atoms and integer primitive translations. Check that " *
            "POSCAR_equlibr is primitive-periodic, uses the same origin, and " *
            "has atom positions consistent with POSCAR_primitive."
        )

    sort!(successful; by=x -> x.max_residual)
    best = first(successful)

    if length(successful) > 1 &&
       isapprox(successful[1].max_residual, successful[2].max_residual; atol=1.0e-12)
        @warn(
            "More than one primitive-origin mapping is symmetry-equivalent. " *
            "Using origin shift $(best.origin_shift)."
        )
    end

    return best.atom_to_primitive, best.atom_to_cell, best.origin_shift, best.max_residual
end

# -----------------------------------------------------------------------------
# Basis mapping
# -----------------------------------------------------------------------------

function normalize_basis_labels(basis_labels, dimension::Int)
    if isnothing(basis_labels)
        return nothing
    end
    labels = String.(vec(basis_labels))
    length(labels) == dimension ||
        error(
            "HAMSTER basis dataset has $(length(labels)) labels, but the " *
            "Hamiltonian dimension is $dimension."
        )
    return labels
end

function build_primitive_basis_map(
    primitive_poscar::AbstractString,
    supercell_poscar::AbstractString,
    hconf_file::AbstractString,
    basis_labels,
    hamiltonian_dimension::Integer;
    lattice_tolerance::Float64=1.0e-7,
    position_tolerance_angstrom::Float64=1.0e-4,
)
    primitive = read_simple_poscar(primitive_poscar)
    supercell = read_simple_poscar(supercell_poscar)
    orbital_tokens = read_hconf_orbitals(hconf_file)

    repetitions, supercell_matrix = infer_diagonal_repetitions(
        primitive.lattice,
        supercell.lattice;
        tolerance=lattice_tolerance,
    )

    atom_to_primitive, atom_to_cell, origin_shift, max_residual = map_supercell_atoms(
        primitive,
        supercell,
        repetitions;
        position_tolerance_angstrom=position_tolerance_angstrom,
    )

    for species in unique(vcat(primitive.atom_species, supercell.atom_species))
        haskey(orbital_tokens, species) ||
            error("No `begin atom $species` orbital block was found in $hconf_file.")
    end

    expected_without_spin = sum(
        length(orbital_tokens[species]) for species in supercell.atom_species
    )

    spin_factor = if hamiltonian_dimension == expected_without_spin
        1
    elseif hamiltonian_dimension == 2 * expected_without_spin
        2
    else
        error(
            "Hamiltonian dimension $hamiltonian_dimension is incompatible with " *
            "$expected_without_spin non-spin orbitals inferred from $hconf_file. " *
            "Expected either $expected_without_spin or $(2 * expected_without_spin)."
        )
    end

    labels = normalize_basis_labels(basis_labels, hamiltonian_dimension)

    primitive_offsets = zeros(Int, length(primitive.atom_species))
    running = 0
    for primitive_atom in eachindex(primitive.atom_species)
        primitive_offsets[primitive_atom] = running
        species = primitive.atom_species[primitive_atom]
        running += spin_factor * length(orbital_tokens[species])
    end
    primitive_dimension = running

    addresses = Vector{PrimitiveOrbitalAddress}(undef, hamiltonian_dimension)
    basis_index = 0

    for super_atom in eachindex(supercell.atom_species)
        species = supercell.atom_species[super_atom]
        tokens = orbital_tokens[species]
        primitive_atom = atom_to_primitive[super_atom]
        cell = atom_to_cell[super_atom]

        for local_orbital in eachindex(tokens)
            for spin_slot in 1:spin_factor
                basis_index += 1
                primitive_orbital = primitive_offsets[primitive_atom] +
                                    (local_orbital - 1) * spin_factor + spin_slot

                synthetic_label = spin_factor == 1 ?
                    "$species-$(tokens[local_orbital])" :
                    "$species-$(tokens[local_orbital])-spin$spin_slot"
                actual_label = isnothing(labels) ? synthetic_label : labels[basis_index]

                startswith(actual_label, "$species-") ||
                    error(
                        "HAMSTER basis label $basis_index is \"$actual_label\", " *
                        "but supercell atom $super_atom is $species. This suggests " *
                        "that the Hamiltonian basis and POSCAR atom order differ."
                    )

                addresses[basis_index] = PrimitiveOrbitalAddress(
                    basis_index,
                    super_atom,
                    primitive_orbital,
                    primitive_atom,
                    local_orbital,
                    tokens[local_orbital],
                    actual_label,
                    spin_slot,
                    cell,
                )
            end
        end
    end

    basis_index == hamiltonian_dimension || error("Internal basis-mapping error.")

    # Every primitive orbital must appear once in every primitive cell.
    keys = [(a.primitive_orbital, a.cell) for a in addresses]
    length(unique(keys)) == length(keys) ||
        error("The inferred (μ,R) basis labels are not unique.")
    length(keys) == primitive_dimension * prod(repetitions) ||
        error("The inferred basis does not fill all primitive-cell copies.")

    println("Primitive/supercell basis mapping:")
    println("  repetitions (n1,n2,n3) = $repetitions")
    println("  primitive cells          = $(prod(repetitions))")
    println("  primitive atoms          = $(length(primitive.atom_species))")
    println("  supercell atoms          = $(length(supercell.atom_species))")
    println("  primitive orbitals       = $primitive_dimension")
    println("  supercell basis size     = $hamiltonian_dimension")
    println("  spin factor              = $spin_factor")
    println("  inferred origin shift    = $origin_shift")
    println("  maximum atom residual Å  = $max_residual")

    return PrimitiveBasisMap(
        repetitions,
        supercell_matrix,
        prod(repetitions),
        length(primitive.atom_species),
        length(supercell.atom_species),
        spin_factor,
        atom_to_primitive,
        atom_to_cell,
        addresses,
    )
end

function relabel_basis_index(mapping::PrimitiveBasisMap, index::Integer)
    checkbounds(mapping.index_to_address, index)
    return mapping.index_to_address[index]
end

function relabel_matrix_element(mapping::PrimitiveBasisMap, i::Integer, j::Integer)
    return (
        row=relabel_basis_index(mapping, i),
        column=relabel_basis_index(mapping, j),
    )
end

function primitive_basis_mapping_table(mapping::PrimitiveBasisMap)
    addresses = mapping.index_to_address
    return (
        basis_index=[a.basis_index for a in addresses],
        supercell_atom=[a.supercell_atom for a in addresses],
        primitive_orbital=[a.primitive_orbital for a in addresses],
        primitive_atom=[a.primitive_atom for a in addresses],
        local_orbital=[a.local_orbital for a in addresses],
        orbital_token=[a.orbital_token for a in addresses],
        basis_label=[a.basis_label for a in addresses],
        spin_slot=[a.spin_slot for a in addresses],
        cell=hcat([collect(a.cell) for a in addresses]...),
    )
end

end # module PrimitiveBasisRelabel
