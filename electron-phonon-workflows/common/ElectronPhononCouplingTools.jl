module ElectronPhononCouplingTools

using HDF5
using LinearAlgebra

include(joinpath(@__DIR__, "VaspTrajectoryStatistics.jl"))
using .VaspTrajectoryStatistics

export load_accumulated_hamiltonian,
       load_displacement_covariance,
       compute_translation_averaged_D,
       save_translation_averaged_D,
       load_translation_averaged_D,
       save_fourier_couplings

const CellVector = NTuple{3, Int}

function read_integer_triplets(dataset)
    raw = Int.(read(dataset))
    ndims(raw) == 2 || error("Expected a two-dimensional integer-triplet dataset.")

    if size(raw, 1) == 3
        return raw
    elseif size(raw, 2) == 3
        return permutedims(raw)
    end

    error("Expected a 3×N or N×3 triplet dataset, found $(size(raw)).")
end

function triplet_column(matrix::Matrix{Int}, index::Integer)::CellVector
    return (matrix[1, index], matrix[2, index], matrix[3, index])
end

function load_accumulated_hamiltonian(filename::AbstractString; load_mean_uH::Bool=true)
    isfile(filename) || error("Accumulated Hamiltonian file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_snapshots",
            "snapshot_indices",
            "hamiltonian_dimension",
            "number_of_atoms",
            "accumulators/mean_u_angstrom",
            "accumulators/mean_H_eV",
            "accumulators/mean_uH_eV_angstrom",
            "electronic/matrix_row_i",
            "electronic/matrix_column_j",
            "electronic/mu",
            "electronic/R",
            "electronic/nu",
            "electronic/Rprime",
            "accumulators/dof_primitive_atom_alpha",
            "accumulators/dof_cell_L",
            "accumulators/dof_component",
            "primitive_basis_mapping/basis_index",
            "primitive_basis_mapping/primitive_orbital_mu",
            "primitive_basis_mapping/cell_R",
            "primitive_basis_mapping/repetitions_n1_n2_n3",
            "primitive_basis_mapping/number_of_primitive_cells",
            "geometry/lattice_angstrom",
        ]
        for path in required
            haskey(f, path) || error("Required dataset is missing: $path")
        end

        snapshot_indices = Vector{Int}(read(f["snapshot_indices"]))
        count = Int(read(f["number_of_snapshots"]))
        count == length(snapshot_indices) ||
            error("number_of_snapshots differs from snapshot_indices length.")

        matrix_i = Vector{Int}(read(f["electronic/matrix_row_i"]))
        matrix_j = Vector{Int}(read(f["electronic/matrix_column_j"]))
        mu = Vector{Int}(read(f["electronic/mu"]))
        nu = Vector{Int}(read(f["electronic/nu"]))
        R = read_integer_triplets(f["electronic/R"])
        Rprime = read_integer_triplets(f["electronic/Rprime"])

        nentries = length(matrix_i)
        all(length(x) == nentries for x in (matrix_j, mu, nu)) ||
            error("Electronic label arrays have inconsistent lengths.")
        size(R, 2) == nentries || error("electronic/R has wrong length.")
        size(Rprime, 2) == nentries || error("electronic/Rprime has wrong length.")

        mean_u = Vector{Float64}(read(f["accumulators/mean_u_angstrom"]))
        mean_H = ComplexF64.(vec(read(f["accumulators/mean_H_eV"])))

        mean_uH_dataset = f["accumulators/mean_uH_eV_angstrom"]
        size(mean_uH_dataset, 2) == nentries ||
            error("mean_uH column count is inconsistent.")
        size(mean_uH_dataset, 1) == length(mean_u) ||
            error("mean_uH row count is inconsistent.")

        # The mean_uH matrix can be many gigabytes.  The D calculation below
        # streams it from HDF5 in electronic-column blocks, so it deliberately
        # calls this loader with load_mean_uH=false.
        mean_uH = load_mean_uH ?
            ComplexF64.(read(mean_uH_dataset)) :
            nothing

        dof_alpha = Vector{Int}(read(f["accumulators/dof_primitive_atom_alpha"]))
        dof_L = read_integer_triplets(f["accumulators/dof_cell_L"])
        dof_component = String.(vec(read(f["accumulators/dof_component"])))
        ndof = length(mean_u)
        length(dof_alpha) == ndof || error("DOF alpha labels have wrong length.")
        size(dof_L, 2) == ndof || error("DOF cell labels have wrong length.")
        length(dof_component) == ndof || error("DOF component labels have wrong length.")

        basis_index = Vector{Int}(read(f["primitive_basis_mapping/basis_index"]))
        basis_mu = Vector{Int}(read(f["primitive_basis_mapping/primitive_orbital_mu"]))
        basis_R = read_integer_triplets(f["primitive_basis_mapping/cell_R"])
        length(basis_index) == length(basis_mu) || error("Basis map arrays differ in length.")
        size(basis_R, 2) == length(basis_index) || error("Basis cell map has wrong length.")

        repetitions_vector = Vector{Int}(read(
            f["primitive_basis_mapping/repetitions_n1_n2_n3"]
        ))
        length(repetitions_vector) == 3 || error("Expected three supercell repetitions.")
        repetitions = Tuple(repetitions_vector)
        number_of_cells = Int(read(
            f["primitive_basis_mapping/number_of_primitive_cells"]
        ))
        number_of_cells == prod(repetitions) ||
            error("Stored primitive-cell count is inconsistent with repetitions.")

        number_of_primitive_atoms = maximum(dof_alpha)
        number_of_primitive_orbitals = maximum(vcat(mu, nu, basis_mu))

        return (
            filename=String(filename),
            number_of_snapshots=count,
            snapshot_indices=snapshot_indices,
            hamiltonian_dimension=Int(read(f["hamiltonian_dimension"])),
            number_of_atoms=Int(read(f["number_of_atoms"])),
            repetitions=repetitions,
            number_of_primitive_cells=number_of_cells,
            number_of_primitive_atoms=number_of_primitive_atoms,
            number_of_primitive_orbitals=number_of_primitive_orbitals,
            supercell_lattice=Matrix{Float64}(read(f["geometry/lattice_angstrom"])),
            mean_u=mean_u,
            mean_H=mean_H,
            mean_uH=mean_uH,
            matrix_i=matrix_i,
            matrix_j=matrix_j,
            mu=mu,
            R=R,
            nu=nu,
            Rprime=Rprime,
            dof_alpha=dof_alpha,
            dof_L=dof_L,
            dof_component=dof_component,
            basis_index=basis_index,
            basis_mu=basis_mu,
            basis_R=basis_R,
        )
    end
end

function load_displacement_covariance(filename::AbstractString)
    isfile(filename) || error("Covariance file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_snapshots",
            "snapshot_indices",
            "covariance/covariance_angstrom2",
            "covariance/mean_u_angstrom",
            "dof/primitive_atom_alpha",
            "dof/cell_L",
            "dof/component",
            "mapping/repetitions_n1_n2_n3",
        ]
        for path in required
            haskey(f, path) || error("Required covariance dataset is missing: $path")
        end

        covariance = Matrix{Float64}(read(f["covariance/covariance_angstrom2"]))
        mean_u = Vector{Float64}(read(f["covariance/mean_u_angstrom"]))
        size(covariance, 1) == size(covariance, 2) ||
            error("Covariance matrix is not square.")
        size(covariance, 1) == length(mean_u) ||
            error("Covariance dimension differs from mean displacement length.")

        dof_alpha = Vector{Int}(read(f["dof/primitive_atom_alpha"]))
        dof_L = read_integer_triplets(f["dof/cell_L"])
        dof_component = String.(vec(read(f["dof/component"])))
        ndof = length(mean_u)
        length(dof_alpha) == ndof || error("Covariance alpha labels have wrong length.")
        size(dof_L, 2) == ndof || error("Covariance cell labels have wrong length.")
        length(dof_component) == ndof || error("Covariance component labels have wrong length.")

        repetitions_vector = Vector{Int}(read(f["mapping/repetitions_n1_n2_n3"]))
        length(repetitions_vector) == 3 || error("Covariance repetitions must have length 3.")

        return (
            filename=String(filename),
            number_of_snapshots=Int(read(f["number_of_snapshots"])),
            snapshot_indices=Vector{Int}(read(f["snapshot_indices"])),
            covariance=covariance,
            mean_u=mean_u,
            dof_alpha=dof_alpha,
            dof_L=dof_L,
            dof_component=dof_component,
            repetitions=Tuple(repetitions_vector),
        )
    end
end

function validate_matching_statistics(accumulated, covariance; tolerance::Float64=1.0e-9)
    # The covariance C and the cross-covariance B=<u ΔH> may be estimated from
    # different numbers of MD samples. In the intended chunked workflow, C can
    # use the full trajectory while B uses only the snapshots for which HAMSTER
    # Hamiltonians were evaluated. Equal sample counts or identical snapshot
    # lists are therefore NOT mathematical requirements for D = C^+ B.
    #
    # What must agree is the coordinate/basis labelling on which the two
    # estimators are represented.
    accumulated.repetitions == covariance.repetitions ||
        error("Hamiltonian and covariance files use different supercells.")
    accumulated.dof_alpha == covariance.dof_alpha ||
        error("Hamiltonian and covariance alpha labels are ordered differently.")
    accumulated.dof_component == covariance.dof_component ||
        error("Hamiltonian and covariance Cartesian labels are ordered differently.")
    accumulated.dof_L == covariance.dof_L ||
        error("Hamiltonian and covariance primitive-cell labels are ordered differently.")

    # Snapshot metadata is retained only as provenance/diagnostics.  It must not
    # prevent the calculation.  In particular, a strided HAMSTER subset is a
    # perfectly valid estimator of B while C is estimated from all MD frames.
    #
    # Likewise, accumulated.mean_u and covariance.mean_u need not coincide:
    # accumulated.mean_u belongs to the Hamiltonian subset and is exactly the
    # mean that must be used in
    #     B = <uH>_H - <u>_H <H>_H.
    # The covariance file has already centered C with its own MD-sample mean.
    return nothing
end

function make_basis_inverse(accumulated)
    inverse = Dict{Tuple{Int, CellVector}, Int}()
    for index in eachindex(accumulated.basis_index)
        key = (
            accumulated.basis_mu[index],
            triplet_column(accumulated.basis_R, index),
        )
        haskey(inverse, key) && error("Duplicate (mu,R) basis address: $key")
        inverse[key] = accumulated.basis_index[index]
    end
    return inverse
end

function make_dof_lookup(accumulated)
    lookup = Dict{Tuple{Int, CellVector, Int}, Int}()
    for dof in eachindex(accumulated.dof_alpha)
        key = (
            accumulated.dof_alpha[dof],
            triplet_column(accumulated.dof_L, dof),
            component_index(accumulated.dof_component[dof]),
        )
        haskey(lookup, key) && error("Duplicate displacement DOF label: $key")
        lookup[key] = dof
    end
    return lookup
end

function make_raw_entry_lookup(accumulated)
    lookup = Dict{Tuple{Int, Int}, Int}()
    for entry in eachindex(accumulated.matrix_i)
        key = (accumulated.matrix_i[entry], accumulated.matrix_j[entry])
        haskey(lookup, key) && error("Duplicate electronic matrix entry: $key")
        lookup[key] = entry
    end
    return lookup
end

function translation_closure_labels(accumulated)
    repetitions = accumulated.repetitions
    cells = enumerate_cells(repetitions)
    labels = Set{Tuple{Int, CellVector, Int, CellVector}}()

    for entry in eachindex(accumulated.mu)
        mu = accumulated.mu[entry]
        nu = accumulated.nu[entry]
        R = triplet_column(accumulated.R, entry)
        Rprime = triplet_column(accumulated.Rprime, entry)

        for shift in cells
            shifted_R = canonical_cell(
                (R[1] - shift[1], R[2] - shift[2], R[3] - shift[3]),
                repetitions,
            )
            shifted_Rprime = canonical_cell(
                (
                    Rprime[1] - shift[1],
                    Rprime[2] - shift[2],
                    Rprime[3] - shift[3],
                ),
                repetitions,
            )
            push!(labels, (mu, shifted_R, nu, shifted_Rprime))
        end
    end

    return sort(
        collect(labels);
        by=x -> (x[1], x[3], x[2][3], x[2][2], x[2][1], x[4][3], x[4][2], x[4][1]),
    )
end

function covariance_eigensystem(
    covariance_matrix::Matrix{Float64};
    relative_cutoff::Float64=1.0e-10,
    absolute_cutoff::Float64=0.0,
)
    symmetrized = Symmetric((covariance_matrix + covariance_matrix') / 2)
    decomposition = eigen(symmetrized)
    eigenvalues = decomposition.values
    eigenvectors = decomposition.vectors

    maximum_eigenvalue = isempty(eigenvalues) ? 0.0 : maximum(abs.(eigenvalues))
    threshold = max(absolute_cutoff, relative_cutoff * maximum_eigenvalue)

    significantly_negative = eigenvalues .< -threshold
    any(significantly_negative) && @warn(
        "The covariance matrix has $(count(significantly_negative)) eigenvalue(s) " *
        "below -$threshold Å². They are discarded in the pseudoinverse."
    )

    inverse_eigenvalues = zeros(Float64, length(eigenvalues))
    retained = falses(length(eigenvalues))
    for index in eachindex(eigenvalues)
        if eigenvalues[index] > threshold
            inverse_eigenvalues[index] = 1.0 / eigenvalues[index]
            retained[index] = true
        end
    end

    return (
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        inverse_eigenvalues=inverse_eigenvalues,
        retained=retained,
        threshold=threshold,
        rank=count(retained),
    )
end

"""
Compute the regression couplings and the primitive-translation average

    D(alpha,a,mu,R,nu,R') = 1/Nc sum_L
        D_raw[(alpha,L,a), (mu,R+L),(nu,R'+L)].

The returned `D_values` has rows `(alpha, Cartesian component)` and columns
`(mu,R,nu,Rprime)`.
"""
function compute_translation_averaged_D(
    accumulated_file::AbstractString,
    covariance_file::AbstractString;
    pseudoinverse_relative_cutoff::Float64=1.0e-10,
    pseudoinverse_absolute_cutoff::Float64=0.0,
    mean_tolerance::Float64=1.0e-9,
    electronic_block_size::Int=512,
    materialize_raw_D::Bool=false,
    show_progress::Bool=true,
)
    electronic_block_size >= 1 ||
        error("electronic_block_size must be at least 1.")

    # IMPORTANT: do not load the huge <uH> matrix here.  It is streamed below.
    accumulated = load_accumulated_hamiltonian(
        accumulated_file;
        load_mean_uH=false,
    )
    covariance = load_displacement_covariance(covariance_file)
    validate_matching_statistics(accumulated, covariance; tolerance=mean_tolerance)

    eigensystem = covariance_eigensystem(
        covariance.covariance;
        relative_cutoff=pseudoinverse_relative_cutoff,
        absolute_cutoff=pseudoinverse_absolute_cutoff,
    )

    retained_indices = findall(eigensystem.retained)
    retained_vectors = eigensystem.eigenvectors[:, retained_indices]
    retained_inverse_eigenvalues =
        eigensystem.inverse_eigenvalues[retained_indices]

    # Build the translation-averaged output labels once.
    dof_lookup = make_dof_lookup(accumulated)
    output_labels = translation_closure_labels(accumulated)
    output_lookup = Dict(
        label => index for (index, label) in enumerate(output_labels)
    )
    cells = enumerate_cells(accumulated.repetitions)
    number_of_cells = accumulated.number_of_primitive_cells

    ionic_labels = [
        (alpha, component)
        for alpha in 1:accumulated.number_of_primitive_atoms
        for component in 1:3
    ]

    D_values = zeros(
        ComplexF64,
        length(ionic_labels),
        length(output_labels),
    )

    nentries = length(accumulated.mean_H)
    ndof = length(accumulated.mean_u)

    # Only allocate the full raw matrix when explicitly requested.  Normally it
    # is not needed at all and would defeat the purpose of the streaming code.
    raw_D = materialize_raw_D ?
        zeros(ComplexF64, ndof, nentries) :
        nothing

    number_of_blocks = cld(nentries, electronic_block_size)

    h5open(accumulated_file, "r") do f
        mean_uH_dataset = f["accumulators/mean_uH_eV_angstrom"]

        size(mean_uH_dataset) == (ndof, nentries) ||
            error(
                "Stored mean_uH dimensions $(size(mean_uH_dataset)) do not " *
                "match expected ($ndof, $nentries)."
            )

        for block_number in 1:number_of_blocks
            first_entry = (block_number - 1) * electronic_block_size + 1
            last_entry = min(
                block_number * electronic_block_size,
                nentries,
            )
            entries = first_entry:last_entry
            nblock = length(entries)

            # Read ONLY this electronic-column slab from HDF5.
            mean_uH_block =
                ComplexF64.(mean_uH_dataset[:, entries])

            # B[:,p] = <u H_p>_H - <u>_H <H_p>_H
            #
            # This allocation is ndof x electronic_block_size, not
            # ndof x all_electronic_entries.
            B_block = mean_uH_block .-
                accumulated.mean_u *
                transpose(accumulated.mean_H[entries])

            # D_block = C^+ B_block, using only retained covariance modes.
            #
            # projected has dimensions rank(C) x block_size.
            projected = retained_vectors' * B_block
            projected .*= reshape(
                retained_inverse_eigenvalues,
                :,
                1,
            )
            D_block = retained_vectors * projected

            if materialize_raw_D
                raw_D[:, entries] .= D_block
            end

            # Directly accumulate the primitive-translation average from this
            # D_block.  This avoids ever constructing full raw_D.
            for local_entry in 1:nblock
                global_entry = first_entry + local_entry - 1

                mu = accumulated.mu[global_entry]
                nu = accumulated.nu[global_entry]
                raw_R = triplet_column(accumulated.R, global_entry)
                raw_Rprime =
                    triplet_column(accumulated.Rprime, global_entry)

                for L in cells
                    # Original definition:
                    #
                    # Dbar(mu,R,nu,R') =
                    #   1/Nc sum_L Draw[(alpha,L),
                    #                  (mu,R+L),(nu,R'+L)].
                    #
                    # Therefore a raw electronic entry at (raw_R,raw_Rprime)
                    # contributes to output (raw_R-L,raw_Rprime-L).
                    output_R = canonical_cell(
                        (
                            raw_R[1] - L[1],
                            raw_R[2] - L[2],
                            raw_R[3] - L[3],
                        ),
                        accumulated.repetitions,
                    )
                    output_Rprime = canonical_cell(
                        (
                            raw_Rprime[1] - L[1],
                            raw_Rprime[2] - L[2],
                            raw_Rprime[3] - L[3],
                        ),
                        accumulated.repetitions,
                    )

                    output_entry = output_lookup[
                        (mu, output_R, nu, output_Rprime)
                    ]

                    for (ionic_row, ionic_label) in enumerate(ionic_labels)
                        alpha, component = ionic_label
                        dof = get(
                            dof_lookup,
                            (alpha, L, component),
                            0,
                        )
                        dof == 0 && error(
                            "Missing ionic DOF " *
                            "(alpha=$alpha,L=$L,a=$component)."
                        )

                        D_values[ionic_row, output_entry] +=
                            D_block[dof, local_entry] / number_of_cells
                    end
                end
            end

            if show_progress &&
               (block_number == 1 ||
                block_number % 20 == 0 ||
                block_number == number_of_blocks)
                println(
                    "D streaming block $block_number/$number_of_blocks: " *
                    "electronic entries $first_entry:$last_entry"
                )
            end
        end
    end

    output_mu = [x[1] for x in output_labels]
    output_R = hcat([collect(x[2]) for x in output_labels]...)
    output_nu = [x[3] for x in output_labels]
    output_Rprime = hcat([collect(x[4]) for x in output_labels]...)

    return (
        accumulated=accumulated,
        covariance=covariance,

        # Full cross_covariance and full raw_D are intentionally not retained.
        # Keeping them was the source of the previous multi-gigabyte memory use.
        cross_covariance=nothing,
        raw_D=raw_D,

        D_values=D_values,
        ionic_alpha=[x[1] for x in ionic_labels],
        ionic_component=[component_label(x[2]) for x in ionic_labels],
        mu=output_mu,
        R=output_R,
        nu=output_nu,
        Rprime=output_Rprime,
        covariance_eigenvalues=eigensystem.eigenvalues,
        covariance_retained_modes=eigensystem.retained,
        covariance_pseudoinverse_threshold=eigensystem.threshold,
        covariance_rank=eigensystem.rank,
    )
end

function save_translation_averaged_D(
    filename::AbstractString,
    result;
    save_raw_D::Bool=false,
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    h5open(temporary, "w") do f
        f["format_version"] = 2

        # Backward-compatible top-level fields: these refer to the Hamiltonian
        # sample used for <H>, <u>, and <uH>.
        f["number_of_snapshots"] = result.accumulated.number_of_snapshots
        f["snapshot_indices"] = result.accumulated.snapshot_indices

        # Explicit provenance for the two statistical estimators entering
        # D = C_MD^+ B_H.
        sampling = create_group(f, "sampling")
        sampling["hamiltonian_number_of_snapshots"] =
            result.accumulated.number_of_snapshots
        sampling["hamiltonian_snapshot_indices"] =
            result.accumulated.snapshot_indices
        sampling["covariance_number_of_snapshots"] =
            result.covariance.number_of_snapshots
        sampling["covariance_snapshot_indices"] =
            result.covariance.snapshot_indices
        sampling["estimator"] =
            "D = C_MD^+ B_H; B_H = <uH>_H - <u>_H <H>_H"

        f["number_of_primitive_cells"] = result.accumulated.number_of_primitive_cells
        f["repetitions_n1_n2_n3"] = collect(result.accumulated.repetitions)
        f["number_of_primitive_atoms"] = result.accumulated.number_of_primitive_atoms
        f["number_of_primitive_orbitals"] = result.accumulated.number_of_primitive_orbitals
        f["supercell_lattice_angstrom"] = result.accumulated.supercell_lattice
        primitive_lattice = result.accumulated.supercell_lattice *
            Diagonal(1.0 ./ collect(result.accumulated.repetitions))
        f["primitive_lattice_angstrom"] = primitive_lattice

        sources = create_group(f, "sources")
        sources["accumulated_hamiltonian"] = abspath(result.accumulated.filename)
        sources["displacement_covariance"] = abspath(result.covariance.filename)

        covariance_group = create_group(f, "covariance_pseudoinverse")
        covariance_group["eigenvalues_angstrom2"] = result.covariance_eigenvalues
        covariance_group["retained_modes"] = Int.(result.covariance_retained_modes)
        covariance_group["threshold_angstrom2"] =
            result.covariance_pseudoinverse_threshold
        covariance_group["rank"] = result.covariance_rank

        ionic = create_group(f, "ionic")
        ionic["primitive_atom_alpha"] = result.ionic_alpha
        ionic["component"] = result.ionic_component

        electronic = create_group(f, "electronic")
        electronic["mu"] = result.mu
        electronic["R"] = result.R
        electronic["nu"] = result.nu
        electronic["Rprime"] = result.Rprime

        couplings = create_group(f, "couplings")
        couplings["D_eV_per_angstrom"] = result.D_values

        if save_raw_D
            isnothing(result.raw_D) && error(
                "save_raw_D=true was requested, but compute_translation_averaged_D " *
                "was run with materialize_raw_D=false. Set materialize_raw_D=true " *
                "only if you intentionally have enough RAM for the full raw D matrix."
            )
            raw = create_group(f, "raw_regression")
            raw["D_alpha_L_eV_per_angstrom"] = result.raw_D
            raw["dof_primitive_atom_alpha"] = result.accumulated.dof_alpha
            raw["dof_cell_L"] = result.accumulated.dof_L
            raw["dof_component"] = result.accumulated.dof_component
            raw["matrix_row_i"] = result.accumulated.matrix_i
            raw["matrix_column_j"] = result.accumulated.matrix_j
        end
    end

    mv(temporary, filename; force=true)
    return nothing
end

function load_translation_averaged_D(filename::AbstractString)
    isfile(filename) || error("D-coupling file does not exist: $filename")

    return h5open(filename, "r") do f
        repetitions_vector = Vector{Int}(read(f["repetitions_n1_n2_n3"]))
        hamiltonian_number_of_snapshots =
            haskey(f, "sampling/hamiltonian_number_of_snapshots") ?
            Int(read(f["sampling/hamiltonian_number_of_snapshots"])) :
            Int(read(f["number_of_snapshots"]))

        hamiltonian_snapshot_indices =
            haskey(f, "sampling/hamiltonian_snapshot_indices") ?
            Vector{Int}(read(f["sampling/hamiltonian_snapshot_indices"])) :
            Vector{Int}(read(f["snapshot_indices"]))

        covariance_number_of_snapshots =
            haskey(f, "sampling/covariance_number_of_snapshots") ?
            Int(read(f["sampling/covariance_number_of_snapshots"])) :
            hamiltonian_number_of_snapshots

        covariance_snapshot_indices =
            haskey(f, "sampling/covariance_snapshot_indices") ?
            Vector{Int}(read(f["sampling/covariance_snapshot_indices"])) :
            hamiltonian_snapshot_indices

        return (
            filename=String(filename),

            # Backward-compatible aliases: Hamiltonian sampling.
            number_of_snapshots=hamiltonian_number_of_snapshots,
            snapshot_indices=hamiltonian_snapshot_indices,

            hamiltonian_number_of_snapshots=hamiltonian_number_of_snapshots,
            hamiltonian_snapshot_indices=hamiltonian_snapshot_indices,
            covariance_number_of_snapshots=covariance_number_of_snapshots,
            covariance_snapshot_indices=covariance_snapshot_indices,

            number_of_primitive_cells=Int(read(f["number_of_primitive_cells"])),
            repetitions=Tuple(repetitions_vector),
            number_of_primitive_atoms=Int(read(f["number_of_primitive_atoms"])),
            number_of_primitive_orbitals=Int(read(f["number_of_primitive_orbitals"])),
            supercell_lattice=Matrix{Float64}(read(f["supercell_lattice_angstrom"])),
            primitive_lattice=Matrix{Float64}(read(f["primitive_lattice_angstrom"])),
            ionic_alpha=Vector{Int}(read(f["ionic/primitive_atom_alpha"])),
            ionic_component=String.(vec(read(f["ionic/component"]))),
            mu=Vector{Int}(read(f["electronic/mu"])),
            R=read_integer_triplets(f["electronic/R"]),
            nu=Vector{Int}(read(f["electronic/nu"])),
            Rprime=read_integer_triplets(f["electronic/Rprime"]),
            D_values=ComplexF64.(read(f["couplings/D_eV_per_angstrom"])),
        )
    end
end

function reduced_phase(integer_k::CellVector, R::CellVector, repetitions::CellVector)
    argument = (
        integer_k[1] * R[1] / repetitions[1] +
        integer_k[2] * R[2] / repetitions[2] +
        integer_k[3] * R[3] / repetitions[3]
    )
    return exp(-2π * im * argument)
end

function save_fourier_couplings(
    filename::AbstractString,
    D_result;
    zero_tolerance::Float64=0.0,
)
    repetitions = D_result.repetitions
    cells = enumerate_cells(repetitions)
    number_of_cells = length(cells)
    number_of_cells == D_result.number_of_primitive_cells ||
        error("D file has an inconsistent primitive-cell count.")

    cell_index = Dict(cell => index for (index, cell) in enumerate(cells))
    number_of_orbitals = D_result.number_of_primitive_orbitals
    number_of_ionic_rows = length(D_result.ionic_alpha)
    number_of_entries = length(D_result.mu)
    size(D_result.D_values) == (number_of_ionic_rows, number_of_entries) ||
        error("D array dimensions are inconsistent with its labels.")

    # Fminus[k,R] = exp(-2π i k·R), where k_i = n_i/N_i.
    Fminus = Matrix{ComplexF64}(undef, number_of_cells, number_of_cells)
    for (k_index, k_integer) in enumerate(cells)
        for (R_index, R) in enumerate(cells)
            Fminus[k_index, R_index] = reduced_phase(k_integer, R, repetitions)
        end
    end
    Fplus = conj.(Fminus)

    # Group electronic columns by primitive orbital pair (mu,nu).
    pair_entries = Dict{Tuple{Int, Int}, Vector{Int}}()
    for entry in 1:number_of_entries
        push!(get!(pair_entries, (D_result.mu[entry], D_result.nu[entry]), Int[]), entry)
    end
    active_pairs = sort(collect(keys(pair_entries)))

    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    block_alpha = Int[]
    block_component = String[]
    block_mu = Int[]
    block_nu = Int[]
    block_dataset = String[]

    h5open(temporary, "w") do f
        f["format_version"] = 2

        # Backward-compatible aliases: Hamiltonian sampling.
        f["number_of_snapshots"] = D_result.number_of_snapshots
        f["snapshot_indices"] = D_result.snapshot_indices

        sampling = create_group(f, "sampling")
        sampling["hamiltonian_number_of_snapshots"] =
            hasproperty(D_result, :hamiltonian_number_of_snapshots) ?
            D_result.hamiltonian_number_of_snapshots :
            D_result.number_of_snapshots
        sampling["hamiltonian_snapshot_indices"] =
            hasproperty(D_result, :hamiltonian_snapshot_indices) ?
            D_result.hamiltonian_snapshot_indices :
            D_result.snapshot_indices
        sampling["covariance_number_of_snapshots"] =
            hasproperty(D_result, :covariance_number_of_snapshots) ?
            D_result.covariance_number_of_snapshots :
            D_result.number_of_snapshots
        sampling["covariance_snapshot_indices"] =
            hasproperty(D_result, :covariance_snapshot_indices) ?
            D_result.covariance_snapshot_indices :
            D_result.snapshot_indices

        f["number_of_primitive_cells"] = number_of_cells
        f["repetitions_n1_n2_n3"] = collect(repetitions)
        f["number_of_primitive_atoms"] = D_result.number_of_primitive_atoms
        f["number_of_primitive_orbitals"] = number_of_orbitals
        f["normalization"] = "1/sqrt(Nc)"
        f["phase_convention"] =
            "exp[-2pi*i*(k+q).R] exp[+2pi*i*k.Rprime]"

        sources = create_group(f, "sources")
        sources["D_couplings"] = abspath(D_result.filename)

        grids = create_group(f, "grids")
        integer_grid = hcat([collect(cell) for cell in cells]...)
        reduced_grid = zeros(Float64, 3, number_of_cells)
        for index in 1:number_of_cells
            reduced_grid[1, index] = integer_grid[1, index] / repetitions[1]
            reduced_grid[2, index] = integer_grid[2, index] / repetitions[2]
            reduced_grid[3, index] = integer_grid[3, index] / repetitions[3]
        end
        grids["k_integer_indices"] = integer_grid
        grids["k_reduced_primitive_reciprocal"] = reduced_grid
        grids["q_integer_indices"] = integer_grid
        grids["q_reduced_primitive_reciprocal"] = reduced_grid
        primitive_reciprocal = 2π .* inv(D_result.primitive_lattice)'
        grids["primitive_reciprocal_lattice_inv_angstrom"] = primitive_reciprocal
        grids["k_cartesian_inv_angstrom"] = primitive_reciprocal * reduced_grid
        grids["q_cartesian_inv_angstrom"] = primitive_reciprocal * reduced_grid
        grids["d_matrix_axis_1"] = "k index"
        grids["d_matrix_axis_2"] = "q index"

        blocks = create_group(f, "d_blocks")
        block_counter = 0

        for ionic_row in 1:number_of_ionic_rows
            alpha = D_result.ionic_alpha[ionic_row]
            component = D_result.ionic_component[ionic_row]

            for pair in active_pairs
                mu, nu = pair
                D_RR = zeros(ComplexF64, number_of_cells, number_of_cells)

                for entry in pair_entries[pair]
                    R = triplet_column(D_result.R, entry)
                    Rprime = triplet_column(D_result.Rprime, entry)
                    D_RR[cell_index[R], cell_index[Rprime]] +=
                        D_result.D_values[ionic_row, entry]
                end

                maximum(abs.(D_RR)) > zero_tolerance || continue

                # M[K,k] = 1/sqrt(Nc) sum_R,R' exp(-i K.R) D(R,R') exp(+i k.R').
                M = (Fminus * D_RR * transpose(Fplus)) / sqrt(number_of_cells)
                d_kq = zeros(ComplexF64, number_of_cells, number_of_cells)

                for (k_index, k_integer) in enumerate(cells)
                    for (q_index, q_integer) in enumerate(cells)
                        K_integer = canonical_cell(k_integer, q_integer, repetitions)
                        K_index = cell_index[K_integer]
                        d_kq[k_index, q_index] = M[K_index, k_index]
                    end
                end

                block_counter += 1
                dataset_name = "block_" * lpad(string(block_counter), 8, '0')
                blocks[dataset_name] = d_kq

                push!(block_alpha, alpha)
                push!(block_component, component)
                push!(block_mu, mu)
                push!(block_nu, nu)
                push!(block_dataset, dataset_name)
            end
        end

        index_group = create_group(f, "block_index")
        index_group["primitive_atom_alpha"] = block_alpha
        index_group["component"] = block_component
        index_group["mu"] = block_mu
        index_group["nu"] = block_nu
        index_group["dataset_name"] = block_dataset
        index_group["number_of_blocks"] = length(block_alpha)
    end

    mv(temporary, filename; force=true)
    return nothing
end

end # module ElectronPhononCouplingTools
