include(joinpath(@__DIR__, "CalculationPaths.jl"))
using .CalculationPaths
const DATA_DIR = stage_data_directory!(ARGS)
const CALC_DIR = DATA_DIR
using HDF5
using LinearAlgebra
using Printf

# =============================================================================
# Phonons from finite-temperature displacement covariance
#
# Inputs (default: same directory as this script)
#   displacement_covariance.h5
#   atomic_masses_amu.txt
#
# Output
#   phonon_modes_from_covariance.h5
#
# Harmonic relation in a translationally periodic system:
#
#   C(q) = k_B T K(q)^+
#
# so, on the supported subspace,
#
#   K(q) = k_B T C(q)^+
#
# The script first projects the noisy supercell covariance onto primitive
# translational symmetry, Fourier transforms it to the commensurate q grid,
# pseudoinverts each small C(q) block, optionally enforces the acoustic sum
# rule at Gamma, constructs the mass-weighted dynamical matrix, and diagonalizes
# it.
#
# q-grid ordering is chosen to match the electron_lattice_d_kq.h5 pipeline:
#   r1 changes fastest, then r2, then r3.
# =============================================================================

const DEFAULT_COVARIANCE_FILE =
    joinpath(DATA_DIR, "displacement_covariance.h5")
const DEFAULT_MASS_FILE =
    (isfile(joinpath(CALC_DIR,"atomic_masses_amu.txt")) ? joinpath(CALC_DIR,"atomic_masses_amu.txt") : joinpath(@__DIR__,"atomic_masses_amu.txt"))
const DEFAULT_OUTPUT_FILE =
    joinpath(DATA_DIR, "phonon_modes_from_covariance.h5")

# Pseudoinverse controls for C(q).
const DEFAULT_PINV_RTOL = 1.0e-10
const DEFAULT_PINV_ATOL_ANGSTROM2 = 0.0

# Enforce exact rigid translations as zero-frequency Gamma modes.
const DEFAULT_ENFORCE_ACOUSTIC_SUM_RULE = true

# Physical constants, exact SI where defined.
const KB_EV_PER_K = 8.617333262145e-5
const EV_J = 1.602176634e-19
const ANGSTROM_M = 1.0e-10
const AMU_KG = 1.66053906660e-27
const HBAR_EV_S = 6.582119569e-16
const C_CM_PER_S = 2.99792458e10

# Conversion:
#   [eV / Angstrom^2 / amu] -> [rad^2 / s^2]
const DYNMAT_TO_OMEGA2_SI =
    EV_J / (ANGSTROM_M^2 * AMU_KG)

# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

function print_help()
    println("""
Usage:
    julia compute_phonons_from_covariance.jl [options]

Options:
    --temperature T
        Temperature in K.

        If omitted, the script asks interactively at startup.

    --covariance FILE
        Covariance HDF5 file.
        Default: $DEFAULT_COVARIANCE_FILE

    --masses FILE
        Text file containing
            ELEMENT  MASS_AMU
        Default: $DEFAULT_MASS_FILE

    --output FILE
        Output HDF5 file.
        Default: $DEFAULT_OUTPUT_FILE

    --pinv-rtol X
        Relative eigenvalue cutoff for pseudoinverting C(q).
        Default: $DEFAULT_PINV_RTOL

    --pinv-atol X
        Absolute covariance-eigenvalue cutoff in Angstrom^2.
        Default: $DEFAULT_PINV_ATOL_ANGSTROM2

    --no-asr
        Do not enforce the acoustic sum rule at Gamma.

    --help
        Show this help.

Notes:
    * Frequencies are calculated only on the q grid commensurate with the
      MD supercell.
    * Negative dynamical-matrix eigenvalues are saved as negative signed
      frequencies, indicating imaginary harmonic modes.
""")
end

function parse_arguments(args)
    settings = Dict{Symbol,Any}(
        :temperature => nothing,
        :covariance => DEFAULT_COVARIANCE_FILE,
        :masses => DEFAULT_MASS_FILE,
        :output => DEFAULT_OUTPUT_FILE,
        :pinv_rtol => DEFAULT_PINV_RTOL,
        :pinv_atol => DEFAULT_PINV_ATOL_ANGSTROM2,
        :enforce_asr => DEFAULT_ENFORCE_ACOUSTIC_SUM_RULE,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)

        elseif arg == "--no-asr"
            settings[:enforce_asr] = false
            i += 1
            continue

        elseif arg in (
            "--temperature",
            "--covariance",
            "--masses",
            "--output",
            "--pinv-rtol",
            "--pinv-atol",
        )
            i == length(args) && error("Missing value after $arg")
            value = args[i + 1]

            if arg == "--temperature"
                settings[:temperature] = parse(Float64, value)
            elseif arg == "--covariance"
                settings[:covariance] = abspath(value)
            elseif arg == "--masses"
                settings[:masses] = abspath(value)
            elseif arg == "--output"
                settings[:output] = abspath(value)
            elseif arg == "--pinv-rtol"
                settings[:pinv_rtol] = parse(Float64, value)
            elseif arg == "--pinv-atol"
                settings[:pinv_atol] = parse(Float64, value)
            end

            i += 2
            continue
        else
            error("Unknown argument: $arg. Use --help.")
        end
    end

    settings[:pinv_rtol] >= 0 ||
        error("--pinv-rtol must be non-negative.")
    settings[:pinv_atol] >= 0 ||
        error("--pinv-atol must be non-negative.")

    return settings
end

function obtain_temperature(settings)
    if !isnothing(settings[:temperature])
        T = settings[:temperature]
    else
        print("Enter temperature in K: ")
        flush(stdout)
        line = strip(readline())
        isempty(line) && error("No temperature was entered.")
        T = parse(Float64, line)
    end

    T > 0 || error("Temperature must be positive.")
    return T
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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
            error("Mass must be positive for species $species.")

        haskey(masses, species) &&
            error("Duplicate mass entry for species $species.")
        masses[species] = mass
    end

    isempty(masses) && error("No masses were read from $filename.")
    return masses
end

function h5_strings(dataset)
    return String.(vec(read(dataset)))
end

function enumerate_cells(repetitions::NTuple{3,Int})
    n1, n2, n3 = repetitions
    return [
        (r1, r2, r3)
        for r3 in 0:(n3 - 1)
        for r2 in 0:(n2 - 1)
        for r1 in 0:(n1 - 1)
    ]
end

canonical_cell(
    x::NTuple{3,Int},
    repetitions::NTuple{3,Int},
) = (
    mod(x[1], repetitions[1]),
    mod(x[2], repetitions[2]),
    mod(x[3], repetitions[3]),
)

cell_difference(
    L2::NTuple{3,Int},
    L1::NTuple{3,Int},
    repetitions::NTuple{3,Int},
) = canonical_cell(
    (L2[1] - L1[1], L2[2] - L1[2], L2[3] - L1[3]),
    repetitions,
)

negative_cell(
    L::NTuple{3,Int},
    repetitions::NTuple{3,Int},
) = canonical_cell((-L[1], -L[2], -L[3]), repetitions)

primitive_dof_index(alpha::Int, component::Int) =
    3 * (alpha - 1) + component

function read_covariance_input(filename::AbstractString)
    isfile(filename) || error("Covariance file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "covariance/covariance_angstrom2",
            "dof/primitive_atom_alpha",
            "dof/cell_L",
            "mapping/repetitions_n1_n2_n3",
            "mapping/number_of_primitive_cells",
            "mapping/number_of_primitive_atoms",
            "mapping/primitive_atom_alpha",
            "geometry/atom_species",
            "geometry/equilibrium_primitive_lattice_angstrom",
            "geometry/equilibrium_primitive_fractional_positions",
        ]

        for name in required
            haskey(f, name) || error(
                "Required dataset '$name' is missing from $filename."
            )
        end

        covariance =
            Matrix{Float64}(read(f["covariance/covariance_angstrom2"]))
        covariance .= (covariance + covariance') ./ 2

        dof_alpha = Vector{Int}(read(f["dof/primitive_atom_alpha"]))
        dof_L = Matrix{Int}(read(f["dof/cell_L"]))

        dof_component = if haskey(f, "dof/component_index")
            Vector{Int}(read(f["dof/component_index"]))
        elseif haskey(f, "dof/component")
            components = h5_strings(f["dof/component"])
            map(c -> c == "x" ? 1 : c == "y" ? 2 :
                     c == "z" ? 3 :
                     error("Unknown Cartesian component '$c'."), components)
        else
            error("Neither dof/component_index nor dof/component exists.")
        end

        repetitions_vector =
            Vector{Int}(read(f["mapping/repetitions_n1_n2_n3"]))
        length(repetitions_vector) == 3 ||
            error("Expected three supercell repetitions.")
        repetitions = Tuple(repetitions_vector)

        number_of_primitive_cells =
            Int(read(f["mapping/number_of_primitive_cells"]))
        number_of_primitive_atoms =
            Int(read(f["mapping/number_of_primitive_atoms"]))

        atom_species = h5_strings(f["geometry/atom_species"])
        atom_alpha =
            Vector{Int}(read(f["mapping/primitive_atom_alpha"]))

        primitive_lattice =
            Matrix{Float64}(
                read(f["geometry/equilibrium_primitive_lattice_angstrom"])
            )
        primitive_fractional =
            Matrix{Float64}(
                read(f["geometry/equilibrium_primitive_fractional_positions"])
            )

        snapshot_indices =
            haskey(f, "snapshot_indices") ?
            Vector{Int}(read(f["snapshot_indices"])) :
            Int[]

        return (
            covariance = covariance,
            dof_alpha = dof_alpha,
            dof_L = dof_L,
            dof_component = dof_component,
            repetitions = repetitions,
            number_of_primitive_cells = number_of_primitive_cells,
            number_of_primitive_atoms = number_of_primitive_atoms,
            atom_species = atom_species,
            atom_alpha = atom_alpha,
            primitive_lattice = primitive_lattice,
            primitive_fractional = primitive_fractional,
            snapshot_indices = snapshot_indices,
        )
    end
end

function infer_primitive_species(data)
    nalpha = data.number_of_primitive_atoms
    species = Vector{String}(undef, nalpha)

    length(data.atom_species) == length(data.atom_alpha) ||
        error("Atom species and atom->alpha arrays have different lengths.")

    for alpha in 1:nalpha
        indices = findall(==(alpha), data.atom_alpha)
        isempty(indices) &&
            error("No supercell atom maps to primitive atom alpha=$alpha.")

        found = unique(data.atom_species[indices])
        length(found) == 1 ||
            error(
                "Primitive atom alpha=$alpha maps to multiple species: " *
                join(found, ", ")
            )
        species[alpha] = only(found)
    end

    return species
end

# -----------------------------------------------------------------------------
# Translational projection of covariance
# -----------------------------------------------------------------------------

function translation_average_covariance(data)
    C = data.covariance
    ndof = length(data.dof_alpha)

    size(C) == (ndof, ndof) ||
        error("Covariance dimensions do not match DOF labels.")
    size(data.dof_L) == (3, ndof) ||
        error("dof/cell_L must have size 3 x ndof.")
    length(data.dof_component) == ndof ||
        error("DOF component labels have wrong length.")

    repetitions = data.repetitions
    cells = enumerate_cells(repetitions)
    Nc = length(cells)

    Nc == data.number_of_primitive_cells ||
        error(
            "Product of repetitions ($Nc) differs from stored primitive-cell count " *
            "($(data.number_of_primitive_cells))."
        )

    cell_index = Dict(cell => i for (i, cell) in enumerate(cells))
    nprim_dof = 3 * data.number_of_primitive_atoms

    # C_delta[i,j,r] =
    #   (1/Nc) sum_L C_{i,L ; j,L+Delta}
    C_delta = zeros(Float64, nprim_dof, nprim_dof, Nc)

    for I in 1:ndof
        alpha = data.dof_alpha[I]
        a = data.dof_component[I]
        LI = (
            data.dof_L[1, I],
            data.dof_L[2, I],
            data.dof_L[3, I],
        )
        i = primitive_dof_index(alpha, a)

        for J in 1:ndof
            beta = data.dof_alpha[J]
            b = data.dof_component[J]
            LJ = (
                data.dof_L[1, J],
                data.dof_L[2, J],
                data.dof_L[3, J],
            )
            j = primitive_dof_index(beta, b)

            delta = cell_difference(LJ, LI, repetitions)
            r = cell_index[delta]
            C_delta[i, j, r] += C[I, J] / Nc
        end
    end

    # Enforce C_ab(Delta) = C_ba(-Delta)^T exactly against roundoff/noise.
    for (r, delta) in enumerate(cells)
        rminus = cell_index[negative_cell(delta, repetitions)]
        A = copy(C_delta[:, :, r])
        B = transpose(C_delta[:, :, rminus])
        average = (A + B) ./ 2
        C_delta[:, :, r] .= average
        C_delta[:, :, rminus] .= transpose(average)
    end

    return C_delta, cells
end

function fourier_covariance(
    C_delta::Array{Float64,3},
    cells::Vector{NTuple{3,Int}},
    repetitions::NTuple{3,Int},
)
    nprim_dof = size(C_delta, 1)
    Nq = length(cells)
    C_q = zeros(ComplexF64, nprim_dof, nprim_dof, Nq)

    for (iq, q_integer) in enumerate(cells)
        current = zeros(ComplexF64, nprim_dof, nprim_dof)

        for (ir, delta) in enumerate(cells)
            phase_argument =
                q_integer[1] * delta[1] / repetitions[1] +
                q_integer[2] * delta[2] / repetitions[2] +
                q_integer[3] * delta[3] / repetitions[3]

            phase = exp(2π * im * phase_argument)
            current .+= phase .* C_delta[:, :, ir]
        end

        # Exact Hermiticity is expected; enforce it numerically.
        C_q[:, :, iq] .= (current + current') ./ 2
    end

    return C_q
end

# -----------------------------------------------------------------------------
# Force constants and phonons
# -----------------------------------------------------------------------------

function pseudoinverse_covariance_block(
    Cq::Matrix{ComplexF64},
    kBT_eV::Float64;
    rtol::Float64,
    atol::Float64,
)
    decomposition = eigen(Hermitian(Cq))
    values = decomposition.values
    vectors = decomposition.vectors

    scale = isempty(values) ? 0.0 : maximum(abs.(values))
    cutoff = max(atol, rtol * scale)

    # A significant negative covariance eigenvalue is not physical.
    significant_negative = findall(x -> x < -cutoff, values)
    isempty(significant_negative) || error(
        "A Fourier covariance block has significantly negative eigenvalues. " *
        "Most negative = $(minimum(values)) Angstrom^2, cutoff = $cutoff. " *
        "Inspect sampling or translational mapping."
    )

    inverse_values = zeros(Float64, length(values))
    retained = falses(length(values))

    for i in eachindex(values)
        if values[i] > cutoff
            inverse_values[i] = 1 / values[i]
            retained[i] = true
        end
    end

    Kq =
        kBT_eV .* (
            vectors * Diagonal(inverse_values) * vectors'
        )
    Kq .= (Kq + Kq') ./ 2

    return Matrix{ComplexF64}(Kq), values, retained, cutoff
end

function enforce_gamma_acoustic_sum_rule!(
    Kq::Matrix{ComplexF64},
    number_of_primitive_atoms::Int,
)
    n = 3 * number_of_primitive_atoms
    size(Kq) == (n, n) ||
        error("Unexpected Gamma force-constant block size.")

    # Uniform Cartesian translations in unweighted displacement coordinates.
    T = zeros(ComplexF64, n, 3)
    for alpha in 1:number_of_primitive_atoms
        for a in 1:3
            T[primitive_dof_index(alpha, a), a] = 1.0
        end
    end

    # Orthogonal projector removing the rigid-translation subspace.
    P = Matrix{ComplexF64}(I, n, n) -
        T * inv(T' * T) * T'

    Kq .= P * Kq * P
    Kq .= (Kq + Kq') ./ 2
    return nothing
end

function calculate_force_constants_q(
    C_q::Array{ComplexF64,3},
    cells::Vector{NTuple{3,Int}},
    temperature_K::Float64;
    rtol::Float64,
    atol::Float64,
    enforce_asr::Bool,
)
    kBT_eV = KB_EV_PER_K * temperature_K
    n = size(C_q, 1)
    Nq = size(C_q, 3)

    K_q = zeros(ComplexF64, n, n, Nq)
    covariance_eigenvalues = zeros(Float64, n, Nq)
    covariance_retained = falses(n, Nq)
    covariance_cutoff = zeros(Float64, Nq)
    covariance_used = copy(C_q)

    gamma_index = findfirst(==((0, 0, 0)), cells)
    isnothing(gamma_index) && error("Could not find Gamma point.")

    for iq in 1:Nq
        Cq = Matrix(C_q[:, :, iq])
        if enforce_asr && iq == gamma_index
            # MD may constrain the mass-weighted centre of mass rather than the
            # unweighted translation. First express covariance in relative
            # (translation-free) coordinates, then invert on that subspace.
            # Projecting only K after inversion does not recover that covariance
            # when the primitive masses differ.
            enforce_gamma_acoustic_sum_rule!(Cq, div(n, 3))
        end
        covariance_used[:, :, iq] .= Cq
        Kq, values, retained, cutoff =
            pseudoinverse_covariance_block(
                Cq,
                kBT_eV;
                rtol=rtol,
                atol=atol,
            )

        if enforce_asr && iq == gamma_index
            enforce_gamma_acoustic_sum_rule!(
                Kq,
                div(n, 3),
            )
        end

        K_q[:, :, iq] .= Kq
        covariance_eigenvalues[:, iq] .= values
        covariance_retained[:, iq] .= retained
        covariance_cutoff[iq] = cutoff
    end

    return (
        K_q = K_q,
        covariance_eigenvalues = covariance_eigenvalues,
        covariance_retained = covariance_retained,
        covariance_cutoff = covariance_cutoff,
        covariance_used = covariance_used,
        kBT_eV = kBT_eV,
    )
end

function inverse_fourier_force_constants(
    K_q::Array{ComplexF64,3},
    cells::Vector{NTuple{3,Int}},
    repetitions::NTuple{3,Int},
)
    n = size(K_q, 1)
    Nq = length(cells)
    Phi = zeros(ComplexF64, n, n, Nq)

    for (ir, delta) in enumerate(cells)
        current = zeros(ComplexF64, n, n)

        for (iq, q_integer) in enumerate(cells)
            phase_argument =
                q_integer[1] * delta[1] / repetitions[1] +
                q_integer[2] * delta[2] / repetitions[2] +
                q_integer[3] * delta[3] / repetitions[3]

            phase = exp(-2π * im * phase_argument)
            current .+= phase .* K_q[:, :, iq]
        end

        Phi[:, :, ir] .= current ./ Nq
    end

    return Phi
end

function reconstruct_supercell_force_constants(data, Phi, cells)
    ndof = length(data.dof_alpha)
    repetitions = data.repetitions
    cell_index = Dict(cell => i for (i, cell) in enumerate(cells))

    K_sc = zeros(ComplexF64, ndof, ndof)

    for I in 1:ndof
        alpha = data.dof_alpha[I]
        a = data.dof_component[I]
        LI = (
            data.dof_L[1, I],
            data.dof_L[2, I],
            data.dof_L[3, I],
        )
        i = primitive_dof_index(alpha, a)

        for J in 1:ndof
            beta = data.dof_alpha[J]
            b = data.dof_component[J]
            LJ = (
                data.dof_L[1, J],
                data.dof_L[2, J],
                data.dof_L[3, J],
            )
            j = primitive_dof_index(beta, b)

            delta = cell_difference(LJ, LI, repetitions)
            r = cell_index[delta]
            K_sc[I, J] = Phi[i, j, r]
        end
    end

    K_sc .= (K_sc + K_sc') ./ 2
    return K_sc
end

function build_mass_arrays(
    primitive_species::Vector{String},
    masses_by_species::Dict{String,Float64},
)
    primitive_masses = zeros(Float64, length(primitive_species))

    for (alpha, species) in enumerate(primitive_species)
        haskey(masses_by_species, species) ||
            error(
                "Mass file has no entry for primitive species '$species'."
            )
        primitive_masses[alpha] = masses_by_species[species]
    end

    mass_dof = repeat(primitive_masses; inner=3)
    return primitive_masses, mass_dof
end

function calculate_phonons(
    K_q::Array{ComplexF64,3},
    primitive_masses_amu::Vector{Float64},
)
    nalpha = length(primitive_masses_amu)
    nmodes = 3 * nalpha
    Nq = size(K_q, 3)

    mass_dof = repeat(primitive_masses_amu; inner=3)
    inv_sqrt_mass = 1.0 ./ sqrt.(mass_dof)

    dynamical_eigenvalues =
        zeros(Float64, nmodes, Nq)
    omega2_si =
        zeros(Float64, nmodes, Nq)
    frequency_thz_signed =
        zeros(Float64, nmodes, Nq)
    frequency_mev_signed =
        zeros(Float64, nmodes, Nq)
    frequency_cm1_signed =
        zeros(Float64, nmodes, Nq)
    is_imaginary =
        falses(nmodes, Nq)

    # e_{alpha a}^{(s)}(q), dimensionless mass-weighted polarization vector.
    polarization =
        zeros(ComplexF64, 3, nalpha, nmodes, Nq)

    # e/sqrt(M), proportional to the actual Cartesian displacement pattern.
    displacement_pattern_amu_mhalf =
        zeros(ComplexF64, 3, nalpha, nmodes, Nq)

    # Same Cartesian pattern rescaled so max atom-vector magnitude is one.
    displacement_pattern_relative =
        zeros(ComplexF64, 3, nalpha, nmodes, Nq)

    for iq in 1:Nq
        Kq = Matrix(K_q[:, :, iq])

        Dq = Diagonal(inv_sqrt_mass) *
             Kq *
             Diagonal(inv_sqrt_mass)
        Dq .= (Dq + Dq') ./ 2

        decomposition = eigen(Hermitian(Dq))
        lambdas = decomposition.values
        evecs = decomposition.vectors

        dynamical_eigenvalues[:, iq] .= lambdas

        for mode in 1:nmodes
            lambda = lambdas[mode]
            omega2 = lambda * DYNMAT_TO_OMEGA2_SI
            omega2_si[mode, iq] = omega2

            omega_abs = sqrt(abs(omega2))
            sign_factor = lambda < 0 ? -1.0 : 1.0

            frequency_thz_signed[mode, iq] =
                sign_factor * omega_abs / (2π * 1.0e12)

            energy_ev = HBAR_EV_S * omega_abs
            frequency_mev_signed[mode, iq] =
                sign_factor * 1000.0 * energy_ev

            frequency_hz = omega_abs / (2π)
            frequency_cm1_signed[mode, iq] =
                sign_factor * frequency_hz / C_CM_PER_S

            is_imaginary[mode, iq] = lambda < 0

            # Fill Cartesian atom representation.
            max_atom_norm = 0.0

            for alpha in 1:nalpha
                for a in 1:3
                    flat = primitive_dof_index(alpha, a)
                    e = evecs[flat, mode]

                    polarization[a, alpha, mode, iq] = e
                    displacement_pattern_amu_mhalf[a, alpha, mode, iq] =
                        e / sqrt(primitive_masses_amu[alpha])
                end

                atom_norm = norm(
                    displacement_pattern_amu_mhalf[:, alpha, mode, iq]
                )
                max_atom_norm = max(max_atom_norm, atom_norm)
            end

            if max_atom_norm > 0
                displacement_pattern_relative[:, :, mode, iq] .=
                    displacement_pattern_amu_mhalf[:, :, mode, iq] ./
                    max_atom_norm
            end
        end
    end

    return (
        dynamical_eigenvalues = dynamical_eigenvalues,
        omega2_si = omega2_si,
        frequency_thz_signed = frequency_thz_signed,
        frequency_mev_signed = frequency_mev_signed,
        frequency_cm1_signed = frequency_cm1_signed,
        is_imaginary = is_imaginary,
        polarization = polarization,
        displacement_pattern_amu_mhalf =
            displacement_pattern_amu_mhalf,
        displacement_pattern_relative =
            displacement_pattern_relative,
    )
end

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

function build_q_grids(cells, repetitions, primitive_lattice)
    Nq = length(cells)
    integer_grid = hcat([collect(q) for q in cells]...)

    reduced_grid = zeros(Float64, 3, Nq)
    for iq in 1:Nq
        reduced_grid[1, iq] =
            integer_grid[1, iq] / repetitions[1]
        reduced_grid[2, iq] =
            integer_grid[2, iq] / repetitions[2]
        reduced_grid[3, iq] =
            integer_grid[3, iq] / repetitions[3]
    end

    primitive_reciprocal =
        2π .* inv(primitive_lattice)'
    q_cartesian =
        primitive_reciprocal * reduced_grid

    return integer_grid, reduced_grid, primitive_reciprocal, q_cartesian
end

function write_output(
    filename,
    settings,
    temperature_K,
    data,
    primitive_species,
    primitive_masses,
    C_delta,
    C_q,
    force_result,
    Phi,
    K_sc,
    cells,
    phonons,
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    repetitions = data.repetitions
    integer_grid,
    reduced_grid,
    primitive_reciprocal,
    q_cartesian =
        build_q_grids(
            cells,
            repetitions,
            data.primitive_lattice,
        )

    max_phi_imag = maximum(abs.(imag.(Phi)))
    max_Ksc_imag = maximum(abs.(imag.(K_sc)))

    h5open(temporary, "w") do f
        f["format_version"] = 1
        f["temperature_K"] = temperature_K
        f["kBT_eV"] = force_result.kBT_eV
        f["number_of_q_points"] = length(cells)
        f["number_of_primitive_atoms"] =
            data.number_of_primitive_atoms
        f["number_of_modes"] =
            3 * data.number_of_primitive_atoms
        f["repetitions_n1_n2_n3"] =
            collect(repetitions)

        sources = create_group(f, "sources")
        sources["covariance_file"] =
            abspath(settings[:covariance])
        sources["mass_file"] =
            abspath(settings[:masses])

        settings_group = create_group(f, "settings")
        settings_group["pseudoinverse_relative_cutoff"] =
            settings[:pinv_rtol]
        settings_group["pseudoinverse_absolute_cutoff_angstrom2"] =
            settings[:pinv_atol]
        settings_group["acoustic_sum_rule_enforced"] =
            settings[:enforce_asr]
        settings_group["gamma_covariance_projected_before_inversion"] = settings[:enforce_asr]

        primitive = create_group(f, "primitive")
        primitive["species"] = primitive_species
        primitive["mass_amu"] = primitive_masses
        primitive["lattice_angstrom"] =
            data.primitive_lattice
        primitive["fractional_positions"] =
            data.primitive_fractional

        grids = create_group(f, "q_grid")
        grids["integer_indices"] = integer_grid
        grids["reduced_primitive_reciprocal"] =
            reduced_grid
        grids["primitive_reciprocal_lattice_inv_angstrom"] =
            primitive_reciprocal
        grids["cartesian_inv_angstrom"] =
            q_cartesian
        grids["ordering"] =
            "r1 fastest, then r2, then r3; matches electron_lattice_d_kq.h5 grid convention"

        covariance_group =
            create_group(f, "covariance")
        covariance_group["translation_averaged_realspace_angstrom2"] =
            C_delta
        covariance_group["q_blocks_angstrom2"] =
            C_q
        covariance_group["q_blocks_used_in_pseudoinverse_angstrom2"] =
            force_result.covariance_used
        covariance_group["q_block_eigendecomposition_source"] =
            "q_blocks_used_in_pseudoinverse_angstrom2; Gamma projected to relative displacements before inversion when ASR enabled"
        covariance_group["q_block_eigenvalues_angstrom2"] =
            force_result.covariance_eigenvalues
        covariance_group["q_block_retained_in_pseudoinverse"] =
            Int8.(force_result.covariance_retained)
        covariance_group["q_block_cutoff_angstrom2"] =
            force_result.covariance_cutoff

        force = create_group(f, "force_constants")
        force["K_q_eV_per_angstrom2"] =
            force_result.K_q
        force["Phi_delta_eV_per_angstrom2"] =
            Phi
        force["delta_cell_integer"] =
            integer_grid
        force["K_supercell_translation_symmetrized_eV_per_angstrom2"] =
            K_sc
        force["max_abs_imag_Phi_eV_per_angstrom2"] =
            max_phi_imag
        force["max_abs_imag_K_supercell_eV_per_angstrom2"] =
            max_Ksc_imag

        phonon = create_group(f, "phonons")
        phonon["dynamical_eigenvalue_eV_per_angstrom2_per_amu"] =
            phonons.dynamical_eigenvalues
        phonon["omega_squared_s_minus2"] =
            phonons.omega2_si
        phonon["frequency_THz_signed"] =
            phonons.frequency_thz_signed
        phonon["energy_meV_signed"] =
            phonons.frequency_mev_signed
        phonon["wavenumber_cm_minus1_signed"] =
            phonons.frequency_cm1_signed
        phonon["is_imaginary"] =
            Int8.(phonons.is_imaginary)

        phonon["polarization_eigenvector"] =
            phonons.polarization
        phonon["polarization_eigenvector_axes"] =
            "component, primitive_atom_alpha, mode, q_index"
        phonon["polarization_normalization"] =
            "sum_alpha,a |e_alpha_a|^2 = 1"

        phonon["cartesian_displacement_pattern_amu_mhalf"] =
            phonons.displacement_pattern_amu_mhalf
        phonon["cartesian_displacement_pattern_amu_mhalf_axes"] =
            "component, primitive_atom_alpha, mode, q_index"
        phonon["cartesian_displacement_pattern_amu_mhalf_definition"] =
            "polarization_eigenvector / sqrt(mass_amu)"

        phonon["cartesian_displacement_pattern_relative"] =
            phonons.displacement_pattern_relative
        phonon["cartesian_displacement_pattern_relative_axes"] =
            "component, primitive_atom_alpha, mode, q_index"
        phonon["cartesian_displacement_pattern_relative_normalization"] =
            "for each mode and q, maximum atom displacement-vector norm = 1"

        if !isempty(data.snapshot_indices)
            f["covariance_snapshot_indices"] =
                data.snapshot_indices
        end
    end

    mv(temporary, filename; force=true)
    return nothing
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    settings = parse_arguments(ARGS)
    temperature_K = obtain_temperature(settings)

    isfile(settings[:covariance]) ||
        error("Required covariance file does not exist: $(settings[:covariance])")
    isfile(settings[:masses]) ||
        error("Required mass file does not exist: $(settings[:masses])")

    println("Temperature:     ", temperature_K, " K")
    println("Covariance:      ", settings[:covariance])
    println("Masses:          ", settings[:masses])
    println("Output:          ", settings[:output])

    data = read_covariance_input(settings[:covariance])
    primitive_species = infer_primitive_species(data)
    masses_by_species = read_mass_file(settings[:masses])
    primitive_masses, _ =
        build_mass_arrays(
            primitive_species,
            masses_by_species,
        )

    println("Primitive species and masses:")
    for alpha in 1:data.number_of_primitive_atoms
        println(
            "  alpha=", alpha,
            "  ", primitive_species[alpha],
            "  ", primitive_masses[alpha], " amu",
        )
    end

    println("Projecting covariance onto primitive translational symmetry...")
    C_delta, cells =
        translation_average_covariance(data)

    println("Fourier transforming covariance to q-space...")
    C_q =
        fourier_covariance(
            C_delta,
            cells,
            data.repetitions,
        )

    println("Constructing effective harmonic force constants...")
    force_result =
        calculate_force_constants_q(
            C_q,
            cells,
            temperature_K;
            rtol=settings[:pinv_rtol],
            atol=settings[:pinv_atol],
            enforce_asr=settings[:enforce_asr],
        )

    println("Transforming force constants back to real space...")
    Phi =
        inverse_fourier_force_constants(
            force_result.K_q,
            cells,
            data.repetitions,
        )

    K_sc =
        reconstruct_supercell_force_constants(
            data,
            Phi,
            cells,
        )

    println("Mass weighting and diagonalizing dynamical matrices...")
    phonons =
        calculate_phonons(
            force_result.K_q,
            primitive_masses,
        )

    write_output(
        settings[:output],
        settings,
        temperature_K,
        data,
        primitive_species,
        primitive_masses,
        C_delta,
        C_q,
        force_result,
        Phi,
        K_sc,
        cells,
        phonons,
    )

    println("Wrote: ", settings[:output])
    println("Done.")
end

main()
