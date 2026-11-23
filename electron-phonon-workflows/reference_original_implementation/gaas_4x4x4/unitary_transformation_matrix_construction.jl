using HDF5
using LinearAlgebra

# Data live separately from the source scripts on hydrogen.
const DATA_DIR = get(ENV, "ELECTRON_PHONON_DATA_DIR", get(ENV, "EP_DATA_DIR", pwd()))

include(joinpath(@__DIR__, "ElectronPhononCouplingTools.jl"))
using .ElectronPhononCouplingTools

# =============================================================================
# Construct orbital -> band unitary transformations U_mu,n(k)
#
# Input:
#   sum_uH_primitive_labels.h5
#
# Optional consistency check:
#   electron_lattice_d_kq.h5
#
# Output:
#   unitary_transformation_matrices.h5
#
# Bloch convention (the SAME electronic convention used in d(k+q,k)):
#
#   |k,mu> = 1/sqrt(Nc) sum_R exp(+2pi*i*k.R) |R,mu>
#
# Therefore
#
#   H_mu,nu(k)
#     = 1/Nc sum_R,R'
#         exp(-2pi*i*k.R)
#         <R,mu|H|R',nu>
#         exp(+2pi*i*k.R')
#
# After primitive-translation averaging,
#
#   h_mu,nu(Delta)
#     = 1/Nc sum_R <R,mu|H|R+Delta,nu>
#
# and
#
#   H_mu,nu(k)
#     = sum_Delta exp(+2pi*i*k.Delta) h_mu,nu(Delta).
#
# For each k:
#
#   H(k) U(k) = U(k) diag(epsilon_n(k))
#
# with columns of U containing the band eigenvectors:
#
#   |n,k> = sum_mu U_mu,n(k) |mu,k>.
# =============================================================================


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

const ACCUMULATED_HAMILTONIAN_FILE =
    joinpath(DATA_DIR, "sum_uH_primitive_labels.h5")

const OUTPUT_FILE =
    joinpath(DATA_DIR, "unitary_transformation_matrices.h5")

# If this file exists, its k-grid is checked against the grid constructed here.
# It is not required for the calculation.
const REFERENCE_d_FILE =
    joinpath(DATA_DIR, "electron_lattice_d_kq.h5")

# Average Hamiltonians should already be Hermitian.  We report the residual and
# then explicitly symmetrize the primitive-periodic representation before
# diagonalization.
const HERMITICITY_WARNING_TOLERANCE_EV = 1.0e-8

# Fix the otherwise arbitrary phase of each nondegenerate eigenvector by making
# its largest-magnitude orbital component real and non-negative.
const APPLY_DETERMINISTIC_PHASE_GAUGE = true


# -----------------------------------------------------------------------------
# Cell / grid helpers
# -----------------------------------------------------------------------------

const CellVector = NTuple{3,Int}

function enumerate_cells(repetitions::CellVector)
    n1, n2, n3 = repetitions

    # Same ordering used in our d(k+q,k) and phonon-grid scripts:
    # r1 changes fastest, then r2, then r3.
    return [
        (r1, r2, r3)
        for r3 in 0:(n3 - 1)
        for r2 in 0:(n2 - 1)
        for r1 in 0:(n1 - 1)
    ]
end

function canonical_cell(
    cell::CellVector,
    repetitions::CellVector,
)::CellVector
    return (
        mod(cell[1], repetitions[1]),
        mod(cell[2], repetitions[2]),
        mod(cell[3], repetitions[3]),
    )
end

function cell_difference(
    Rprime::CellVector,
    R::CellVector,
    repetitions::CellVector,
)::CellVector
    return canonical_cell(
        (
            Rprime[1] - R[1],
            Rprime[2] - R[2],
            Rprime[3] - R[3],
        ),
        repetitions,
    )
end

function negative_cell(
    cell::CellVector,
    repetitions::CellVector,
)::CellVector
    return canonical_cell(
        (-cell[1], -cell[2], -cell[3]),
        repetitions,
    )
end

function triplet_column(
    matrix::AbstractMatrix{<:Integer},
    index::Integer,
)::CellVector
    return (
        Int(matrix[1, index]),
        Int(matrix[2, index]),
        Int(matrix[3, index]),
    )
end

function positive_bloch_phase(
    integer_k::CellVector,
    R::CellVector,
    repetitions::CellVector,
)
    argument =
        integer_k[1] * R[1] / repetitions[1] +
        integer_k[2] * R[2] / repetitions[2] +
        integer_k[3] * R[3] / repetitions[3]

    return exp(+2π * im * argument)
end


# -----------------------------------------------------------------------------
# Primitive orbital metadata
# -----------------------------------------------------------------------------

function read_integer_triplets_local(dataset)
    raw = Int.(read(dataset))
    ndims(raw) == 2 ||
        error("Expected a two-dimensional integer-triplet dataset.")

    if size(raw, 1) == 3
        return raw
    elseif size(raw, 2) == 3
        return permutedims(raw)
    end

    error("Expected a 3xN or Nx3 triplet dataset, found $(size(raw)).")
end

function read_primitive_orbital_metadata(
    filename::AbstractString,
    number_of_orbitals::Int,
)
    return h5open(filename, "r") do f
        group = "primitive_basis_mapping"

        required = [
            "$group/primitive_orbital_mu",
            "$group/primitive_atom",
            "$group/local_orbital",
            "$group/orbital_token",
            "$group/basis_label",
            "$group/spin_slot",
            "$group/cell_R",
        ]

        for path in required
            haskey(f, path) ||
                error("Required orbital metadata dataset is missing: $path")
        end

        mu_all =
            Vector{Int}(read(f["$group/primitive_orbital_mu"]))
        alpha_all =
            Vector{Int}(read(f["$group/primitive_atom"]))
        local_all =
            Vector{Int}(read(f["$group/local_orbital"]))
        token_all =
            String.(vec(read(f["$group/orbital_token"])))
        label_all =
            String.(vec(read(f["$group/basis_label"])))
        spin_all =
            Vector{Int}(read(f["$group/spin_slot"]))
        R_all =
            read_integer_triplets_local(f["$group/cell_R"])

        n = length(mu_all)

        all(length(x) == n for x in (
            alpha_all,
            local_all,
            token_all,
            label_all,
            spin_all,
        )) || error("Primitive basis metadata arrays have inconsistent lengths.")

        size(R_all, 2) == n ||
            error("primitive_basis_mapping/cell_R has inconsistent length.")

        alpha = zeros(Int, number_of_orbitals)
        local_orbital = zeros(Int, number_of_orbitals)
        orbital_token = fill("", number_of_orbitals)
        basis_label = fill("", number_of_orbitals)
        spin_slot = zeros(Int, number_of_orbitals)

        found = falses(number_of_orbitals)

        # Prefer the representative in primitive cell R=(0,0,0).  If a given
        # mu is not present there for some unexpected reason, use its first copy.
        for prefer_zero_cell in (true, false)
            for i in 1:n
                mu = mu_all[i]
                1 <= mu <= number_of_orbitals ||
                    error("Invalid primitive orbital mu=$mu in basis mapping.")

                found[mu] && continue

                R = triplet_column(R_all, i)
                if prefer_zero_cell && R != (0, 0, 0)
                    continue
                end

                alpha[mu] = alpha_all[i]
                local_orbital[mu] = local_all[i]
                orbital_token[mu] = token_all[i]
                basis_label[mu] = label_all[i]
                spin_slot[mu] = spin_all[i]
                found[mu] = true
            end
        end

        all(found) ||
            error(
                "Could not infer metadata for primitive orbital(s): " *
                join(findall(.!found), ", ")
            )

        return (
            primitive_atom_alpha = alpha,
            local_orbital = local_orbital,
            orbital_token = orbital_token,
            basis_label = basis_label,
            spin_slot = spin_slot,
        )
    end
end


# -----------------------------------------------------------------------------
# Primitive translational average of <H>
# -----------------------------------------------------------------------------

"""
Construct

    h[mu,nu,Delta]
      = (1/Nc) sum_R <R,mu| <H> |R+Delta,nu>

from the sparse union stored in sum_uH_primitive_labels.h5.

Missing sparse entries are implicitly zero, so dividing by Nc implements the
primitive translation average including translated copies that are zero in the
stored sparse pattern.

The returned h is then explicitly projected onto Hermiticity:

    h_mu,nu(Delta)
      = conj( h_nu,mu(-Delta) ).
"""
function construct_translation_averaged_hamiltonian(accumulated)
    repetitions = accumulated.repetitions
    cells = enumerate_cells(repetitions)
    Nc = length(cells)

    Nc == accumulated.number_of_primitive_cells ||
        error(
            "Stored primitive-cell count differs from the product of repetitions."
        )

    cell_index = Dict(cell => i for (i, cell) in enumerate(cells))

    norb = accumulated.number_of_primitive_orbitals
    h = zeros(ComplexF64, norb, norb, Nc)

    nentries = length(accumulated.mean_H)

    all(length(x) == nentries for x in (
        accumulated.mu,
        accumulated.nu,
    )) || error("Mean-H labels have inconsistent lengths.")

    size(accumulated.R, 2) == nentries ||
        error("electronic/R length differs from mean_H.")
    size(accumulated.Rprime, 2) == nentries ||
        error("electronic/Rprime length differs from mean_H.")

    for entry in 1:nentries
        mu = accumulated.mu[entry]
        nu = accumulated.nu[entry]

        R = triplet_column(accumulated.R, entry)
        Rprime = triplet_column(accumulated.Rprime, entry)
        delta = cell_difference(Rprime, R, repetitions)

        h[mu, nu, cell_index[delta]] +=
            accumulated.mean_H[entry] / Nc
    end

    # Measure the Hermiticity mismatch before projection.
    max_hermiticity_residual = 0.0

    for (idelta, delta) in enumerate(cells)
        ineg = cell_index[negative_cell(delta, repetitions)]

        residual =
            h[:, :, idelta] -
            adjoint(h[:, :, ineg])

        max_hermiticity_residual =
            max(
                max_hermiticity_residual,
                maximum(abs.(residual)),
            )
    end

    # Enforce the real-space Hermiticity relation exactly.  Process each
    # Delta/-Delta pair only once.
    visited = falses(Nc)

    for (idelta, delta) in enumerate(cells)
        visited[idelta] && continue

        ineg = cell_index[negative_cell(delta, repetitions)]

        A = copy(h[:, :, idelta])
        Bdagger = adjoint(h[:, :, ineg])

        averaged = (A + Bdagger) ./ 2

        h[:, :, idelta] .= averaged
        h[:, :, ineg] .= adjoint(averaged)

        visited[idelta] = true
        visited[ineg] = true
    end

    return (
        h = h,
        cells = cells,
        max_hermiticity_residual_eV = max_hermiticity_residual,
    )
end


# -----------------------------------------------------------------------------
# Fourier transform and diagonalization
# -----------------------------------------------------------------------------

function construct_Hk(
    h::Array{ComplexF64,3},
    cells::Vector{CellVector},
    repetitions::CellVector,
)
    norb = size(h, 1)
    size(h, 2) == norb ||
        error("Real-space primitive Hamiltonian is not square in orbital indices.")
    size(h, 3) == length(cells) ||
        error("Real-space Hamiltonian cell dimension is inconsistent.")

    Nk = length(cells)
    Hk = zeros(ComplexF64, norb, norb, Nk)

    # From
    #
    # |k,mu> = 1/sqrt(Nc) sum_R exp(+2pi i k.R)|R,mu>
    #
    # we obtain
    #
    # H_mu,nu(k) = sum_Delta exp(+2pi i k.Delta) h_mu,nu(Delta).
    for (ik, integer_k) in enumerate(cells)
        block = zeros(ComplexF64, norb, norb)

        for (idelta, delta) in enumerate(cells)
            block .+=
                positive_bloch_phase(
                    integer_k,
                    delta,
                    repetitions,
                ) .* h[:, :, idelta]
        end

        # This should already be Hermitian after the real-space projection.
        Hk[:, :, ik] .= (block + block') ./ 2
    end

    return Hk
end

function fix_eigenvector_phase!(
    vector::AbstractVector{ComplexF64},
)
    isempty(vector) && return nothing

    pivot = argmax(abs.(vector))
    value = vector[pivot]

    abs(value) == 0 && return nothing

    phase = value / abs(value)
    vector ./= phase

    # Remove an irrelevant tiny imaginary residue at the pivot caused by
    # floating-point arithmetic.
    vector[pivot] = ComplexF64(abs(vector[pivot]), 0.0)

    return nothing
end

function diagonalize_Hk(
    Hk::Array{ComplexF64,3};
    deterministic_phase_gauge::Bool=true,
)
    norb = size(Hk, 1)
    Nk = size(Hk, 3)

    energies = zeros(Float64, norb, Nk)
    U = zeros(ComplexF64, norb, norb, Nk)

    unitarity_error = zeros(Float64, Nk)
    diagonalization_residual = zeros(Float64, Nk)
    Hk_hermiticity_error = zeros(Float64, Nk)

    identity_orbital = Matrix{ComplexF64}(I, norb, norb)

    for ik in 1:Nk
        H = Matrix(Hk[:, :, ik])

        Hk_hermiticity_error[ik] =
            maximum(abs.(H - H'))

        decomposition = eigen(Hermitian(H))

        energies[:, ik] .= decomposition.values
        U[:, :, ik] .= decomposition.vectors

        if deterministic_phase_gauge
            for band in 1:norb
                fix_eigenvector_phase!(
                    view(U, :, band, ik)
                )
            end
        end

        Uk = Matrix(U[:, :, ik])

        unitarity_error[ik] =
            maximum(abs.(Uk' * Uk - identity_orbital))

        diagonalization_residual[ik] =
            maximum(
                abs.(
                    H * Uk -
                    Uk * Diagonal(energies[:, ik])
                )
            )
    end

    return (
        energies_eV = energies,
        U = U,
        unitarity_error = unitarity_error,
        diagonalization_residual_eV =
            diagonalization_residual,
        Hk_hermiticity_error_eV =
            Hk_hermiticity_error,
    )
end


# -----------------------------------------------------------------------------
# Optional validation against d-file grid
# -----------------------------------------------------------------------------

function validate_grid_against_d_file(
    filename::AbstractString,
    cells::Vector{CellVector},
    repetitions::CellVector,
)
    isfile(filename) || return false

    h5open(filename, "r") do f
        haskey(f, "grids/k_integer_indices") ||
            error(
                "Reference d-file exists but has no grids/k_integer_indices: $filename"
            )

        stored = Int.(read(f["grids/k_integer_indices"]))

        if size(stored, 1) != 3 && size(stored, 2) == 3
            stored = permutedims(stored)
        end

        expected = hcat([collect(cell) for cell in cells]...)

        stored == expected ||
            error(
                "Constructed k-grid does not match electron_lattice_d_kq.h5. " *
                "This would make the orbital->band transformation inconsistent " *
                "with d(k+q,k)."
            )

        if haskey(f, "repetitions_n1_n2_n3")
            d_repetitions =
                Tuple(Vector{Int}(read(f["repetitions_n1_n2_n3"])))

            d_repetitions == repetitions ||
                error(
                    "Reference d-file uses repetitions $d_repetitions but " *
                    "the averaged Hamiltonian uses $repetitions."
                )
        end
    end

    return true
end


# -----------------------------------------------------------------------------
# HDF5 output
# -----------------------------------------------------------------------------

function save_result(
    filename::AbstractString,
    accumulated,
    orbital_metadata,
    translation_result,
    Hk,
    diagonalization,
    reference_d_checked::Bool,
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    repetitions = accumulated.repetitions
    cells = translation_result.cells
    Nk = length(cells)
    norb = accumulated.number_of_primitive_orbitals

    integer_grid =
        hcat([collect(cell) for cell in cells]...)

    reduced_grid = zeros(Float64, 3, Nk)
    for ik in 1:Nk
        reduced_grid[1, ik] =
            integer_grid[1, ik] / repetitions[1]
        reduced_grid[2, ik] =
            integer_grid[2, ik] / repetitions[2]
        reduced_grid[3, ik] =
            integer_grid[3, ik] / repetitions[3]
    end

    primitive_lattice =
        accumulated.supercell_lattice *
        Diagonal(1.0 ./ collect(repetitions))

    primitive_reciprocal =
        2π .* inv(primitive_lattice)'

    k_cartesian =
        primitive_reciprocal * reduced_grid

    h5open(temporary, "w") do f
        f["format_version"] = 1

        f["number_of_snapshots"] =
            accumulated.number_of_snapshots
        f["snapshot_indices"] =
            accumulated.snapshot_indices

        f["number_of_primitive_cells"] =
            accumulated.number_of_primitive_cells
        f["number_of_primitive_orbitals"] =
            norb
        f["number_of_k_points"] =
            Nk
        f["repetitions_n1_n2_n3"] =
            collect(repetitions)

        f["bloch_basis_definition"] =
            "|k,mu> = 1/sqrt(Nc) sum_R exp(+2pi*i*k.R) |R,mu>"
        f["hamiltonian_fourier_convention"] =
            "H_mu_nu(k) = sum_Delta exp(+2pi*i*k.Delta) h_mu_nu(Delta)"
        f["band_state_definition"] =
            "|n,k> = sum_mu U_mu_n(k) |mu,k>"
        f["diagonalization_definition"] =
            "H(k) U(k) = U(k) diag(epsilon_n(k))"
        f["band_ordering"] =
            "ascending eigenvalue at each k independently"
        f["eigenvector_phase_gauge"] =
            APPLY_DETERMINISTIC_PHASE_GAUGE ?
            "largest-magnitude orbital component made real and non-negative" :
            "raw LAPACK eigenvector phases"
        f["degeneracy_note"] =
            "Within exactly or nearly degenerate subspaces the eigenvectors remain non-unique up to a unitary rotation."

        sources = create_group(f, "sources")
        sources["accumulated_hamiltonian"] =
            abspath(ACCUMULATED_HAMILTONIAN_FILE)
        sources["reference_d_file"] =
            isfile(REFERENCE_d_FILE) ?
            abspath(REFERENCE_d_FILE) :
            ""
        sources["reference_d_grid_checked"] =
            reference_d_checked

        grids = create_group(f, "grids")
        grids["k_integer_indices"] =
            integer_grid
        grids["k_reduced_primitive_reciprocal"] =
            reduced_grid
        grids["primitive_lattice_angstrom"] =
            primitive_lattice
        grids["primitive_reciprocal_lattice_inv_angstrom"] =
            primitive_reciprocal
        grids["k_cartesian_inv_angstrom"] =
            k_cartesian
        grids["ordering"] =
            "r1 fastest, then r2, then r3; same convention as electron_lattice_d_kq.h5"

        orbitals = create_group(f, "orbitals")
        orbitals["mu"] = collect(1:norb)
        orbitals["primitive_atom_alpha"] =
            orbital_metadata.primitive_atom_alpha
        orbitals["local_orbital"] =
            orbital_metadata.local_orbital
        orbitals["orbital_token"] =
            orbital_metadata.orbital_token
        orbitals["basis_label"] =
            orbital_metadata.basis_label
        orbitals["spin_slot"] =
            orbital_metadata.spin_slot

        realspace = create_group(f, "realspace_hamiltonian")
        realspace["delta_cell_integer"] =
            integer_grid
        realspace["h_mu_nu_delta_eV"] =
            translation_result.h
        realspace["axes"] =
            "mu, nu, delta_index"
        realspace["definition"] =
            "h_mu_nu(Delta) = (1/Nc) sum_R <R,mu|<H>|R+Delta,nu>"
        realspace["max_preprojection_hermiticity_residual_eV"] =
            translation_result.max_hermiticity_residual_eV

        momentum = create_group(f, "momentum_hamiltonian")
        momentum["H_k_eV"] = Hk
        momentum["axes"] = "mu, nu, k_index"
        momentum["max_hermiticity_error_eV"] =
            diagonalization.Hk_hermiticity_error_eV

        bands = create_group(f, "bands")
        bands["energy_eV"] =
            diagonalization.energies_eV
        bands["energy_axes"] =
            "band, k_index"

        bands["U_mu_n_k"] =
            diagonalization.U
        bands["U_axes"] =
            "mu, band, k_index"

        bands["unitarity_error_max_abs"] =
            diagonalization.unitarity_error
        bands["diagonalization_residual_eV_max_abs"] =
            diagonalization.diagonalization_residual_eV
    end

    mv(temporary, filename; force=true)
    return nothing
end


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    isfile(ACCUMULATED_HAMILTONIAN_FILE) ||
        error(
            "Accumulated Hamiltonian file does not exist: " *
            ACCUMULATED_HAMILTONIAN_FILE
        )

    println("Loading averaged Hamiltonian...")
    accumulated =
        load_accumulated_hamiltonian(
            ACCUMULATED_HAMILTONIAN_FILE;
            load_mean_uH=false,
        )

    println(
        "Primitive orbitals: ",
        accumulated.number_of_primitive_orbitals,
    )
    println(
        "Primitive cells in supercell: ",
        accumulated.number_of_primitive_cells,
    )
    println(
        "Hamiltonian snapshots averaged: ",
        accumulated.number_of_snapshots,
    )

    orbital_metadata =
        read_primitive_orbital_metadata(
            ACCUMULATED_HAMILTONIAN_FILE,
            accumulated.number_of_primitive_orbitals,
        )

    println("Projecting <H> onto primitive translational symmetry...")
    translation_result =
        construct_translation_averaged_hamiltonian(
            accumulated
        )

    residual =
        translation_result.max_hermiticity_residual_eV

    println(
        "Maximum pre-projection real-space Hermiticity residual: ",
        residual,
        " eV",
    )

    if residual > HERMITICITY_WARNING_TOLERANCE_EV
        @warn(
            "The averaged Hamiltonian has a larger-than-expected Hermiticity " *
            "residual before explicit projection: $residual eV."
        )
    end

    reference_d_checked =
        validate_grid_against_d_file(
            REFERENCE_d_FILE,
            translation_result.cells,
            accumulated.repetitions,
        )

    if reference_d_checked
        println(
            "Verified k-grid ordering against: ",
            REFERENCE_d_FILE,
        )
    else
        println(
            "Reference d-file not found; using the same analytically defined " *
            "grid ordering without external validation."
        )
    end

    println("Fourier transforming <H> to H(k)...")
    Hk =
        construct_Hk(
            translation_result.h,
            translation_result.cells,
            accumulated.repetitions,
        )

    println("Diagonalizing H(k) at every k point...")
    diagonalization =
        diagonalize_Hk(
            Hk;
            deterministic_phase_gauge =
                APPLY_DETERMINISTIC_PHASE_GAUGE,
        )

    println(
        "Maximum unitarity error: ",
        maximum(diagonalization.unitarity_error),
    )
    println(
        "Maximum diagonalization residual: ",
        maximum(
            diagonalization.diagonalization_residual_eV
        ),
        " eV",
    )

    save_result(
        OUTPUT_FILE,
        accumulated,
        orbital_metadata,
        translation_result,
        Hk,
        diagonalization,
        reference_d_checked,
    )

    println("Saved orbital-to-band unitaries: ", OUTPUT_FILE)
    println("Done.")

    return nothing
end

main()
