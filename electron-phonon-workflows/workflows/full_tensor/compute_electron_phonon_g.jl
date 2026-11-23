const CALC_DIR = ENV["EP_CALC_DIR"]
const DATA_DIR = ENV["EP_DATA_DIR"]
using HDF5
using LinearAlgebra
using Printf

# =============================================================================
# Electron-phonon coupling from d(k+q,k) and covariance-derived phonons
#
# Inputs (default: same directory as this script)
#   electron_lattice_d_kq.h5
#   phonon_modes_from_covariance.h5
#   atomic_masses_amu.txt
#
# Output
#   electron_phonon_g_kq.h5
#
# Formula:
#
#   g^(s)_{mu,nu}(q,k)
#     = sum_{alpha,a}
#         sqrt[hbar / (2 M_alpha omega_s(q))]
#         e^(s)_{alpha a}(q)
#         d^(alpha a)_{mu,nu}(k+q,k)
#
# d is assumed to be in eV/Angstrom.
# The zero-point displacement factor is converted to Angstrom, therefore
# g is saved in eV.
#
# The input d datasets have axes:
#   axis 1 = k index
#   axis 2 = q index
#
# The output g datasets use exactly the same convention.
#
# Imaginary phonons and modes with |f| below MIN_FREQUENCY_THz do not possess
# a finite real harmonic zero-point amplitude. Their g(:,q) columns are saved
# as complex NaN and an explicit validity mask is written to the output.
# =============================================================================

const DEFAULT_D_FILE =
    joinpath(DATA_DIR, "electron_lattice_d_kq.h5")
const DEFAULT_PHONON_FILE =
    joinpath(DATA_DIR, "phonon_modes_from_covariance.h5")
const DEFAULT_MASS_FILE =
    (isfile(joinpath(CALC_DIR,"atomic_masses_amu.txt")) ? joinpath(CALC_DIR,"atomic_masses_amu.txt") : joinpath(@__DIR__,"atomic_masses_amu.txt"))
const DEFAULT_OUTPUT_FILE =
    joinpath(DATA_DIR, "electron_phonon_g_kq.h5")

const DEFAULT_MIN_FREQUENCY_THz = 1.0e-6

# Physical constants.
const HBAR_J_S = 1.054571817e-34
const AMU_KG = 1.66053906660e-27
const ANGSTROM_M = 1.0e-10

# -----------------------------------------------------------------------------
# Command-line handling
# -----------------------------------------------------------------------------

function print_help()
    println("""
Usage:
    julia compute_electron_phonon_g.jl [options]

Options:
    --d-file FILE
        Momentum-space deformation-coupling file.
        Default: $DEFAULT_D_FILE

    --phonons FILE
        Phonon file produced by compute_phonons_from_covariance.jl.
        Default: $DEFAULT_PHONON_FILE

    --masses FILE
        Text file containing
            ELEMENT  MASS_AMU
        Default: $DEFAULT_MASS_FILE

    --output FILE
        Output HDF5 file.
        Default: $DEFAULT_OUTPUT_FILE

    --min-frequency-thz X
        Modes with frequency <= X THz are marked invalid for the harmonic
        1/sqrt(omega) prefactor.
        Default: $DEFAULT_MIN_FREQUENCY_THz

    --help
        Show this message.

Output organization:
    /g_blocks/block_XXXXXXXX
        One complex Nk x Nq matrix for a fixed (mode, mu, nu).

    /block_index/mode
    /block_index/mu
    /block_index/nu
    /block_index/dataset_name

    /phonons/valid_for_g
        Boolean-like Int8 array with shape (mode, q).

For invalid (mode,q) points the corresponding entire k-column of g is NaN.
""")
end

function parse_arguments(args)
    settings = Dict{Symbol,Any}(
        :d_file => DEFAULT_D_FILE,
        :phonon_file => DEFAULT_PHONON_FILE,
        :mass_file => DEFAULT_MASS_FILE,
        :output => DEFAULT_OUTPUT_FILE,
        :min_frequency_thz => DEFAULT_MIN_FREQUENCY_THz,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)

        elseif arg in (
            "--d-file",
            "--phonons",
            "--masses",
            "--output",
            "--min-frequency-thz",
        )
            i == length(args) && error("Missing value after $arg")
            value = args[i + 1]

            if arg == "--d-file"
                settings[:d_file] = abspath(value)
            elseif arg == "--phonons"
                settings[:phonon_file] = abspath(value)
            elseif arg == "--masses"
                settings[:mass_file] = abspath(value)
            elseif arg == "--output"
                settings[:output] = abspath(value)
            elseif arg == "--min-frequency-thz"
                settings[:min_frequency_thz] = parse(Float64, value)
            end

            i += 2
            continue
        else
            error("Unknown command-line argument: $arg. Use --help.")
        end
    end

    settings[:min_frequency_thz] >= 0 ||
        error("--min-frequency-thz must be non-negative.")

    return settings
end

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

function h5_strings(dataset)
    return String.(vec(read(dataset)))
end

function read_mass_file(filename::AbstractString)
    isfile(filename) || error("Mass file does not exist: $filename")

    masses = Dict{String,Float64}()

    for (line_number, raw_line) in enumerate(eachline(filename))
        line = strip(first(split(String(raw_line), '#'; limit=2)))
        isempty(line) && continue

        fields = split(line)
        length(fields) == 2 ||
            error(
                "Invalid mass-file line $line_number in $filename. " *
                "Expected: ELEMENT MASS_AMU"
            )

        species = String(fields[1])
        mass = parse(Float64, fields[2])
        mass > 0 ||
            error("Mass for species $species must be positive.")

        haskey(masses, species) &&
            error("Duplicate mass entry for species $species.")

        masses[species] = mass
    end

    isempty(masses) &&
        error("No atomic masses were found in $filename.")

    return masses
end

function component_index(component::AbstractString)
    c = lowercase(strip(String(component)))
    c == "x" && return 1
    c == "y" && return 2
    c == "z" && return 3
    error("Unknown Cartesian component '$component'. Expected x, y, or z.")
end

function require_dataset(f, name::AbstractString, filename::AbstractString)
    haskey(f, name) ||
        error("Required dataset '$name' is missing from $filename.")
end

# -----------------------------------------------------------------------------
# Read phonon data
# -----------------------------------------------------------------------------

function load_phonon_data(filename::AbstractString)
    isfile(filename) || error("Phonon file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_q_points",
            "number_of_primitive_atoms",
            "number_of_modes",
            "repetitions_n1_n2_n3",
            "primitive/species",
            "primitive/mass_amu",
            "q_grid/integer_indices",
            "q_grid/reduced_primitive_reciprocal",
            "q_grid/cartesian_inv_angstrom",
            "phonons/omega_squared_s_minus2",
            "phonons/frequency_THz_signed",
            "phonons/is_imaginary",
            "phonons/polarization_eigenvector",
        ]

        for name in required
            require_dataset(f, name, filename)
        end

        repetitions_vector =
            Vector{Int}(read(f["repetitions_n1_n2_n3"]))
        length(repetitions_vector) == 3 ||
            error("Phonon file must contain three supercell repetitions.")

        species = h5_strings(f["primitive/species"])
        masses_amu =
            Vector{Float64}(read(f["primitive/mass_amu"]))

        number_of_q_points =
            Int(read(f["number_of_q_points"]))
        number_of_primitive_atoms =
            Int(read(f["number_of_primitive_atoms"]))
        number_of_modes =
            Int(read(f["number_of_modes"]))

        q_integer =
            Matrix{Int}(read(f["q_grid/integer_indices"]))
        q_reduced =
            Matrix{Float64}(
                read(f["q_grid/reduced_primitive_reciprocal"])
            )
        q_cartesian =
            Matrix{Float64}(
                read(f["q_grid/cartesian_inv_angstrom"])
            )

        omega2 =
            Matrix{Float64}(
                read(f["phonons/omega_squared_s_minus2"])
            )
        frequency_thz =
            Matrix{Float64}(
                read(f["phonons/frequency_THz_signed"])
            )
        is_imaginary =
            Int8.(read(f["phonons/is_imaginary"])) .!= 0

        polarization =
            ComplexF64.(
                read(f["phonons/polarization_eigenvector"])
            )

        size(omega2) == (number_of_modes, number_of_q_points) ||
            error("Unexpected omega_squared array dimensions.")
        size(frequency_thz) == (number_of_modes, number_of_q_points) ||
            error("Unexpected frequency array dimensions.")
        size(is_imaginary) == (number_of_modes, number_of_q_points) ||
            error("Unexpected imaginary-mode mask dimensions.")

        size(polarization) ==
            (
                3,
                number_of_primitive_atoms,
                number_of_modes,
                number_of_q_points,
            ) ||
            error(
                "Unexpected polarization eigenvector dimensions. Expected " *
                "(3, Nprimitive_atoms, Nmodes, Nq)."
            )

        length(species) == number_of_primitive_atoms ||
            error(
                "primitive/species length does not match " *
                "number_of_primitive_atoms."
            )
        length(masses_amu) == number_of_primitive_atoms ||
            error(
                "primitive/mass_amu length does not match " *
                "number_of_primitive_atoms."
            )

        return (
            number_of_q_points = number_of_q_points,
            number_of_primitive_atoms = number_of_primitive_atoms,
            number_of_modes = number_of_modes,
            repetitions = Tuple(repetitions_vector),
            species = species,
            masses_amu = masses_amu,
            q_integer = q_integer,
            q_reduced = q_reduced,
            q_cartesian = q_cartesian,
            omega2 = omega2,
            frequency_thz = frequency_thz,
            is_imaginary = is_imaginary,
            polarization = polarization,
            temperature_K =
                haskey(f, "temperature_K") ?
                Float64(read(f["temperature_K"])) :
                NaN,
            energy_mev =
                haskey(f, "phonons/energy_meV_signed") ?
                Matrix{Float64}(read(f["phonons/energy_meV_signed"])) :
                nothing,
            wavenumber_cm1 =
                haskey(f, "phonons/wavenumber_cm_minus1_signed") ?
                Matrix{Float64}(
                    read(f["phonons/wavenumber_cm_minus1_signed"])
                ) :
                nothing,
        )
    end
end

# -----------------------------------------------------------------------------
# Read d-file metadata
# -----------------------------------------------------------------------------

function load_d_metadata(filename::AbstractString)
    isfile(filename) || error("d-coupling file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_primitive_cells",
            "repetitions_n1_n2_n3",
            "number_of_primitive_atoms",
            "number_of_primitive_orbitals",
            "grids/k_integer_indices",
            "grids/k_reduced_primitive_reciprocal",
            "grids/q_integer_indices",
            "grids/q_reduced_primitive_reciprocal",
            "grids/k_cartesian_inv_angstrom",
            "grids/q_cartesian_inv_angstrom",
            "block_index/primitive_atom_alpha",
            "block_index/component",
            "block_index/mu",
            "block_index/nu",
            "block_index/dataset_name",
            "block_index/number_of_blocks",
        ]

        for name in required
            require_dataset(f, name, filename)
        end

        repetitions_vector =
            Vector{Int}(read(f["repetitions_n1_n2_n3"]))
        length(repetitions_vector) == 3 ||
            error("d-file must contain three supercell repetitions.")

        return (
            number_of_primitive_cells =
                Int(read(f["number_of_primitive_cells"])),
            number_of_primitive_atoms =
                Int(read(f["number_of_primitive_atoms"])),
            number_of_primitive_orbitals =
                Int(read(f["number_of_primitive_orbitals"])),
            repetitions = Tuple(repetitions_vector),

            k_integer =
                Matrix{Int}(read(f["grids/k_integer_indices"])),
            k_reduced =
                Matrix{Float64}(
                    read(f["grids/k_reduced_primitive_reciprocal"])
                ),
            q_integer =
                Matrix{Int}(read(f["grids/q_integer_indices"])),
            q_reduced =
                Matrix{Float64}(
                    read(f["grids/q_reduced_primitive_reciprocal"])
                ),
            k_cartesian =
                Matrix{Float64}(
                    read(f["grids/k_cartesian_inv_angstrom"])
                ),
            q_cartesian =
                Matrix{Float64}(
                    read(f["grids/q_cartesian_inv_angstrom"])
                ),

            block_alpha =
                Vector{Int}(
                    read(f["block_index/primitive_atom_alpha"])
                ),
            block_component =
                h5_strings(f["block_index/component"]),
            block_mu =
                Vector{Int}(read(f["block_index/mu"])),
            block_nu =
                Vector{Int}(read(f["block_index/nu"])),
            block_dataset =
                h5_strings(f["block_index/dataset_name"]),
            number_of_blocks =
                Int(read(f["block_index/number_of_blocks"])),

            normalization =
                haskey(f, "normalization") ?
                String(read(f["normalization"])) :
                "",
            phase_convention =
                haskey(f, "phase_convention") ?
                String(read(f["phase_convention"])) :
                "",
        )
    end
end

# -----------------------------------------------------------------------------
# Validation and phonon zero-point prefactors
# -----------------------------------------------------------------------------

function validate_inputs(
    d,
    phonons,
    masses_by_species;
    mass_tolerance::Float64 = 1.0e-8,
)
    d.number_of_primitive_atoms ==
        phonons.number_of_primitive_atoms ||
        error(
            "d-file and phonon file use different numbers of primitive atoms: " *
            "$(d.number_of_primitive_atoms) vs " *
            "$(phonons.number_of_primitive_atoms)."
        )

    d.repetitions == phonons.repetitions ||
        error(
            "d-file and phonon file use different supercell repetitions: " *
            "$(d.repetitions) vs $(phonons.repetitions)."
        )

    size(d.q_integer, 2) ==
        phonons.number_of_q_points ||
        error("d-file and phonon file have different q-grid sizes.")

    d.q_integer == phonons.q_integer ||
        error(
            "The integer q grids in d-file and phonon file differ. " *
            "The electron-phonon contraction requires identical q ordering."
        )

    isapprox(
        d.q_reduced,
        phonons.q_reduced;
        rtol = 0.0,
        atol = 1.0e-12,
    ) || error(
        "The reduced q grids in d-file and phonon file differ."
    )

    for alpha in 1:phonons.number_of_primitive_atoms
        species = phonons.species[alpha]

        haskey(masses_by_species, species) ||
            error(
                "Mass file has no entry for primitive species '$species'."
            )

        input_mass = masses_by_species[species]
        phonon_mass = phonons.masses_amu[alpha]

        isapprox(
            input_mass,
            phonon_mass;
            rtol = mass_tolerance,
            atol = 0.0,
        ) || error(
            "Mass mismatch for alpha=$alpha ($species): " *
            "mass file = $input_mass amu, " *
            "phonon file = $phonon_mass amu."
        )
    end

    nblocks = d.number_of_blocks
    for array in (
        d.block_alpha,
        d.block_component,
        d.block_mu,
        d.block_nu,
        d.block_dataset,
    )
        length(array) == nblocks ||
            error("d-file block-index arrays have inconsistent lengths.")
    end

    for alpha in d.block_alpha
        1 <= alpha <= phonons.number_of_primitive_atoms ||
            error("d-file contains invalid primitive atom alpha=$alpha.")
    end

    for mu in d.block_mu
        1 <= mu <= d.number_of_primitive_orbitals ||
            error("d-file contains invalid orbital mu=$mu.")
    end

    for nu in d.block_nu
        1 <= nu <= d.number_of_primitive_orbitals ||
            error("d-file contains invalid orbital nu=$nu.")
    end

    return nothing
end

function primitive_masses_from_file(phonons, masses_by_species)
    return [
        masses_by_species[phonons.species[alpha]]
        for alpha in 1:phonons.number_of_primitive_atoms
    ]
end

function build_valid_mode_mask(
    phonons;
    min_frequency_thz::Float64,
)
    nmodes = phonons.number_of_modes
    nq = phonons.number_of_q_points

    valid = falses(nmodes, nq)
    near_zero = falses(nmodes, nq)

    for mode in 1:nmodes
        for iq in 1:nq
            freq = phonons.frequency_thz[mode, iq]

            near_zero[mode, iq] =
                abs(freq) <= min_frequency_thz

            valid[mode, iq] =
                !phonons.is_imaginary[mode, iq] &&
                phonons.omega2[mode, iq] > 0.0 &&
                freq > min_frequency_thz
        end
    end

    return valid, near_zero
end

function build_displacement_factors_angstrom(
    phonons,
    primitive_masses_amu,
    valid,
)
    nalpha = phonons.number_of_primitive_atoms
    nmodes = phonons.number_of_modes
    nq = phonons.number_of_q_points

    # factor[a, alpha, mode, q] =
    #   sqrt(hbar / (2 M_alpha omega_mode(q))) * e[a,alpha,mode,q]
    #
    # Units: Angstrom.
    factor =
        zeros(ComplexF64, 3, nalpha, nmodes, nq)

    for alpha in 1:nalpha
        mass_kg =
            primitive_masses_amu[alpha] * AMU_KG

        for mode in 1:nmodes
            for iq in 1:nq
                valid[mode, iq] || continue

                omega =
                    sqrt(phonons.omega2[mode, iq])

                zero_point_amplitude_angstrom =
                    sqrt(
                        HBAR_J_S /
                        (2.0 * mass_kg * omega)
                    ) / ANGSTROM_M

                for a in 1:3
                    factor[a, alpha, mode, iq] =
                        zero_point_amplitude_angstrom *
                        phonons.polarization[
                            a,
                            alpha,
                            mode,
                            iq,
                        ]
                end
            end
        end
    end

    return factor
end

# -----------------------------------------------------------------------------
# Block organization
# -----------------------------------------------------------------------------

function group_d_blocks_by_orbital_pair(d)
    pair_entries =
        Dict{Tuple{Int,Int},Vector{Int}}()

    seen =
        Set{Tuple{Int,Int,Int,Int}}()

    for entry in 1:d.number_of_blocks
        mu = d.block_mu[entry]
        nu = d.block_nu[entry]
        alpha = d.block_alpha[entry]
        a = component_index(d.block_component[entry])

        key = (mu, nu, alpha, a)
        key in seen &&
            error(
                "Duplicate d block for (mu,nu,alpha,component)=" *
                "($mu,$nu,$alpha,$a)."
            )
        push!(seen, key)

        push!(
            get!(pair_entries, (mu, nu), Int[]),
            entry,
        )
    end

    active_pairs = sort(collect(keys(pair_entries)))
    return pair_entries, active_pairs
end

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

function write_electron_phonon_couplings(
    settings,
    d,
    phonons,
    primitive_masses_amu,
    valid_mode_q,
    near_zero_mode_q,
    displacement_factor,
)
    pair_entries, active_pairs =
        group_d_blocks_by_orbital_pair(d)

    nk = size(d.k_integer, 2)
    nq = size(d.q_integer, 2)
    nmodes = phonons.number_of_modes

    temporary = String(settings[:output]) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    nan_complex = ComplexF64(NaN, NaN)

    h5open(settings[:d_file], "r") do fd
        h5open(temporary, "w") do fo

            # -------------------------------------------------------------
            # Global metadata
            # -------------------------------------------------------------

            fo["format_version"] = 1
            # Preserve the MD sampling provenance of the input d tensor.
            for key in ("snapshot_indices", "number_of_snapshots")
                haskey(fd, key) && (fo[key] = read(fd[key]))
            end
            fo["number_of_k_points"] = nk
            fo["number_of_q_points"] = nq
            fo["number_of_modes"] = nmodes
            fo["number_of_primitive_atoms"] =
                phonons.number_of_primitive_atoms
            fo["number_of_primitive_orbitals"] =
                d.number_of_primitive_orbitals
            if isfinite(phonons.temperature_K)
                fo["temperature_K"] = phonons.temperature_K
            end
            fo["repetitions_n1_n2_n3"] =
                collect(d.repetitions)

            fo["units"] = "eV"
            fo["formula"] =
                "g_s_munu(q,k) = sum_alpha,a sqrt(hbar/(2 M_alpha omega_s(q))) * e_s_alpha_a(q) * d_alpha_a_munu(k+q,k)"
            fo["polarization_conjugated"] = false
            fo["g_matrix_axis_1"] = "k index"
            fo["g_matrix_axis_2"] = "q index"

            sources = create_group(fo, "sources")
            sources["d_couplings"] =
                abspath(settings[:d_file])
            sources["phonons"] =
                abspath(settings[:phonon_file])
            sources["mass_file"] =
                abspath(settings[:mass_file])

            calculation = create_group(fo, "settings")
            calculation["minimum_frequency_THz"] =
                settings[:min_frequency_thz]
            calculation["invalid_mode_q_policy"] =
                "g(:,q) = complex NaN for imaginary or near-zero harmonic modes"

            # -------------------------------------------------------------
            # Primitive-cell metadata
            # -------------------------------------------------------------

            primitive = create_group(fo, "primitive")
            primitive["species"] = phonons.species
            primitive["mass_amu"] =
                primitive_masses_amu

            # -------------------------------------------------------------
            # Grids: copied from d-file, because g uses exactly its k/q layout
            # -------------------------------------------------------------

            grids = create_group(fo, "grids")
            grids["k_integer_indices"] =
                d.k_integer
            grids["k_reduced_primitive_reciprocal"] =
                d.k_reduced
            grids["q_integer_indices"] =
                d.q_integer
            grids["q_reduced_primitive_reciprocal"] =
                d.q_reduced
            grids["k_cartesian_inv_angstrom"] =
                d.k_cartesian
            grids["q_cartesian_inv_angstrom"] =
                d.q_cartesian
            grids["g_matrix_axis_1"] = "k index"
            grids["g_matrix_axis_2"] = "q index"

            # -------------------------------------------------------------
            # Phonon information needed to interpret the mode index
            # -------------------------------------------------------------

            phonon_group = create_group(fo, "phonons")
            phonon_group["frequency_THz_signed"] =
                phonons.frequency_thz
            phonon_group["omega_squared_s_minus2"] =
                phonons.omega2
            phonon_group["is_imaginary"] =
                Int8.(phonons.is_imaginary)
            phonon_group["is_near_zero"] =
                Int8.(near_zero_mode_q)
            phonon_group["valid_for_g"] =
                Int8.(valid_mode_q)
            phonon_group["polarization_eigenvector"] =
                phonons.polarization
            phonon_group["polarization_eigenvector_axes"] =
                "component, primitive_atom_alpha, mode, q_index"

            if !isnothing(phonons.energy_mev)
                phonon_group["energy_meV_signed"] =
                    phonons.energy_mev
            end

            if !isnothing(phonons.wavenumber_cm1)
                phonon_group["wavenumber_cm_minus1_signed"] =
                    phonons.wavenumber_cm1
            end

            # This is useful for checking the contraction explicitly.
            phonon_group["zero_point_displacement_factor_angstrom"] =
                displacement_factor
            phonon_group["zero_point_displacement_factor_axes"] =
                "component, primitive_atom_alpha, mode, q_index"
            phonon_group["zero_point_displacement_factor_definition"] =
                "sqrt(hbar/(2 M_alpha omega_s(q))) * polarization_eigenvector"

            # -------------------------------------------------------------
            # g blocks
            # -------------------------------------------------------------

            g_blocks = create_group(fo, "g_blocks")

            block_mode = Int[]
            block_mu = Int[]
            block_nu = Int[]
            block_dataset = String[]

            block_counter = 0

            total_pairs = length(active_pairs)

            for (pair_number, pair) in enumerate(active_pairs)
                mu, nu = pair
                entries = pair_entries[pair]

                # Load only the d^{alpha a}_{mu nu}(k,q) matrices needed for
                # this one orbital pair. They are reused for all phonon modes.
                pair_d = Vector{
                    NamedTuple{
                        (:alpha, :component, :values),
                        Tuple{Int,Int,Matrix{ComplexF64}}
                    }
                }()

                for entry in entries
                    alpha = d.block_alpha[entry]
                    a =
                        component_index(
                            d.block_component[entry]
                        )
                    dataset_name =
                        d.block_dataset[entry]

                    path =
                        "d_blocks/" * dataset_name
                    haskey(fd, path) ||
                        error(
                            "d-file index refers to missing dataset '$path'."
                        )

                    values =
                        ComplexF64.(read(fd[path]))

                    size(values) == (nk, nq) ||
                        error(
                            "Unexpected d-block dimensions for '$path': " *
                            "$(size(values)); expected ($nk,$nq)."
                        )

                    push!(
                        pair_d,
                        (
                            alpha = alpha,
                            component = a,
                            values = values,
                        ),
                    )
                end

                for mode in 1:nmodes
                    g = zeros(ComplexF64, nk, nq)

                    for term in pair_d
                        alpha = term.alpha
                        a = term.component
                        d_kq = term.values

                        for iq in 1:nq
                            valid_mode_q[mode, iq] ||
                                continue

                            coefficient =
                                displacement_factor[
                                    a,
                                    alpha,
                                    mode,
                                    iq,
                                ]

                            @views g[:, iq] .+=
                                coefficient .* d_kq[:, iq]
                        end
                    end

                    # Mark invalid harmonic quantization points explicitly.
                    for iq in 1:nq
                        if !valid_mode_q[mode, iq]
                            @views fill!(g[:, iq], nan_complex)
                        end
                    end

                    block_counter += 1
                    dataset_name =
                        "block_" *
                        lpad(string(block_counter), 8, '0')

                    g_blocks[dataset_name] = g

                    push!(block_mode, mode)
                    push!(block_mu, mu)
                    push!(block_nu, nu)
                    push!(block_dataset, dataset_name)
                end

                println(
                    "Processed orbital pair ",
                    pair_number,
                    "/",
                    total_pairs,
                    ": (mu,nu)=(",
                    mu,
                    ",",
                    nu,
                    ")",
                )
            end

            index_group = create_group(fo, "block_index")
            index_group["mode"] = block_mode
            index_group["mu"] = block_mu
            index_group["nu"] = block_nu
            index_group["dataset_name"] =
                block_dataset
            index_group["number_of_blocks"] =
                block_counter
            index_group["number_of_active_orbital_pairs"] =
                length(active_pairs)

            # Preserve useful d-convention metadata.
            conventions = create_group(fo, "conventions")
            conventions["d_normalization"] =
                d.normalization
            conventions["d_phase_convention"] =
                d.phase_convention
            conventions["phonon_factor_phase_convention"] =
                "phonon polarization used exactly as stored; no complex conjugation"
        end
    end

    mv(temporary, settings[:output]; force=true)
    return nothing
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    settings = parse_arguments(ARGS)

    for filename in (
        settings[:d_file],
        settings[:phonon_file],
        settings[:mass_file],
    )
        isfile(filename) ||
            error("Required input file does not exist: $filename")
    end

    println("d couplings:     ", settings[:d_file])
    println("phonons:         ", settings[:phonon_file])
    println("masses:          ", settings[:mass_file])
    println("output:          ", settings[:output])
    println(
        "minimum phonon frequency for finite g: ",
        settings[:min_frequency_thz],
        " THz",
    )

    masses_by_species =
        read_mass_file(settings[:mass_file])
    phonons =
        load_phonon_data(settings[:phonon_file])
    d =
        load_d_metadata(settings[:d_file])

    validate_inputs(
        d,
        phonons,
        masses_by_species,
    )

    primitive_masses_amu =
        primitive_masses_from_file(
            phonons,
            masses_by_species,
        )

    println("Primitive atoms:")
    for alpha in 1:phonons.number_of_primitive_atoms
        println(
            "  alpha=",
            alpha,
            "  species=",
            phonons.species[alpha],
            "  mass=",
            primitive_masses_amu[alpha],
            " amu",
        )
    end

    valid_mode_q, near_zero_mode_q =
        build_valid_mode_mask(
            phonons;
            min_frequency_thz =
                settings[:min_frequency_thz],
        )

    number_valid = count(valid_mode_q)
    number_total = length(valid_mode_q)
    println(
        "Valid harmonic (mode,q) points: ",
        number_valid,
        "/",
        number_total,
    )

    displacement_factor =
        build_displacement_factors_angstrom(
            phonons,
            primitive_masses_amu,
            valid_mode_q,
        )

    println("Contracting d with phonon modes...")
    write_electron_phonon_couplings(
        settings,
        d,
        phonons,
        primitive_masses_amu,
        valid_mode_q,
        near_zero_mode_q,
        displacement_factor,
    )

    println("Wrote: ", settings[:output])
    println("Done.")
    return nothing
end

main()
