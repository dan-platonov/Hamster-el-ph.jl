using HDF5
using LinearAlgebra

include(joinpath(@__DIR__, "CalculationPaths.jl"))
using .CalculationPaths
const DATA_DIR = stage_data_directory!(ARGS; path_only=true)

# =============================================================================
# Transform electron-phonon couplings from orbital basis to band basis
#
# Inputs:
#   electron_phonon_g_kq.h5
#   unitary_transformation_matrices.h5
#
# Output:
#   electron_phonon_g_band_kq.h5
#
# Orbital-basis input:
#
#   g^s_{mu,nu}(k+q,k)
#
# Band-basis transformation:
#
#   g^s_{m,n}(k+q,k)
#     = sum_{mu,nu}
#         U^*_{mu,m}(k+q)
#         g^s_{mu,nu}(k+q,k)
#         U_{nu,n}(k)
#
# i.e.
#
#   g^s_band(k+q,k)
#     = U(k+q)' * g^s_orbital(k+q,k) * U(k)
#
# The input orbital-g HDF5 file stores one Nk x Nq block for every
# (mode, mu, nu).  The output uses the SAME block organization, but with
# (mode, band_m, band_n).
#
# Memory strategy:
#   The full band-basis tensor is NOT materialized in RAM.
#   One (mode,q) slice is reconstructed and transformed at a time, and the
#   resulting k-column is written directly into the corresponding HDF5 blocks.
# =============================================================================


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

const ORBITAL_G_FILE =
    joinpath(DATA_DIR, "electron_phonon_g_kq.h5")

const UNITARY_FILE =
    joinpath(DATA_DIR, "unitary_transformation_matrices.h5")

const OUTPUT_FILE =
    joinpath(DATA_DIR, "electron_phonon_g_band_kq.h5")

const SHOW_PROGRESS = false


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

const CellVector = NTuple{3,Int}

function h5_strings(dataset)
    return String.(vec(read(dataset)))
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

function add_cells(
    k::CellVector,
    q::CellVector,
    repetitions::CellVector,
)::CellVector
    return canonical_cell(
        (
            k[1] + q[1],
            k[2] + q[2],
            k[3] + q[3],
        ),
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

function normalized_integer_grid(raw)
    grid = Int.(raw)

    ndims(grid) == 2 ||
        error("Expected a two-dimensional integer k/q grid.")

    if size(grid, 1) == 3
        return grid
    elseif size(grid, 2) == 3
        return permutedims(grid)
    end

    error(
        "Expected a 3xN or Nx3 integer k/q grid, found $(size(grid))."
    )
end

function require_dataset(
    f,
    path::AbstractString,
    filename::AbstractString,
)
    haskey(f, path) ||
        error("Required dataset '$path' is missing from $filename.")
end


# -----------------------------------------------------------------------------
# Load orbital-basis g metadata
# -----------------------------------------------------------------------------

function load_orbital_g_metadata(filename::AbstractString)
    isfile(filename) ||
        error("Orbital-basis g file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_k_points",
            "number_of_q_points",
            "number_of_modes",
            "number_of_primitive_orbitals",
            "repetitions_n1_n2_n3",
            "grids/k_integer_indices",
            "grids/q_integer_indices",
            "grids/k_reduced_primitive_reciprocal",
            "grids/q_reduced_primitive_reciprocal",
            "grids/k_cartesian_inv_angstrom",
            "grids/q_cartesian_inv_angstrom",
            "block_index/mode",
            "block_index/mu",
            "block_index/nu",
            "block_index/dataset_name",
            "block_index/number_of_blocks",
        ]

        for path in required
            require_dataset(f, path, filename)
        end

        repetitions_vector =
            Vector{Int}(read(f["repetitions_n1_n2_n3"]))

        length(repetitions_vector) == 3 ||
            error("Expected three supercell repetitions.")

        block_mode =
            Vector{Int}(read(f["block_index/mode"]))
        block_mu =
            Vector{Int}(read(f["block_index/mu"]))
        block_nu =
            Vector{Int}(read(f["block_index/nu"]))
        block_dataset =
            h5_strings(f["block_index/dataset_name"])

        number_of_blocks =
            Int(read(f["block_index/number_of_blocks"]))

        all(length(x) == number_of_blocks for x in (
            block_mode,
            block_mu,
            block_nu,
            block_dataset,
        )) || error(
            "Orbital-g block-index arrays have inconsistent lengths."
        )

        return (
            filename = String(filename),

            number_of_k_points =
                Int(read(f["number_of_k_points"])),
            number_of_q_points =
                Int(read(f["number_of_q_points"])),
            number_of_modes =
                Int(read(f["number_of_modes"])),
            number_of_primitive_orbitals =
                Int(read(f["number_of_primitive_orbitals"])),

            repetitions = Tuple(repetitions_vector),

            k_integer =
                normalized_integer_grid(
                    read(f["grids/k_integer_indices"])
                ),
            q_integer =
                normalized_integer_grid(
                    read(f["grids/q_integer_indices"])
                ),

            k_reduced =
                Matrix{Float64}(
                    read(
                        f[
                            "grids/k_reduced_primitive_reciprocal"
                        ]
                    )
                ),
            q_reduced =
                Matrix{Float64}(
                    read(
                        f[
                            "grids/q_reduced_primitive_reciprocal"
                        ]
                    )
                ),

            k_cartesian =
                Matrix{Float64}(
                    read(f["grids/k_cartesian_inv_angstrom"])
                ),
            q_cartesian =
                Matrix{Float64}(
                    read(f["grids/q_cartesian_inv_angstrom"])
                ),

            block_mode = block_mode,
            block_mu = block_mu,
            block_nu = block_nu,
            block_dataset = block_dataset,
            number_of_blocks = number_of_blocks,

            valid_for_g =
                haskey(f, "phonons/valid_for_g") ?
                Int8.(read(f["phonons/valid_for_g"])) .!= 0 :
                nothing,

            frequency_THz =
                haskey(f, "phonons/frequency_THz_signed") ?
                Matrix{Float64}(
                    read(f["phonons/frequency_THz_signed"])
                ) :
                nothing,

            omega2 =
                haskey(f, "phonons/omega_squared_s_minus2") ?
                Matrix{Float64}(
                    read(f["phonons/omega_squared_s_minus2"])
                ) :
                nothing,

            is_imaginary =
                haskey(f, "phonons/is_imaginary") ?
                Int8.(read(f["phonons/is_imaginary"])) .!= 0 :
                nothing,

            is_near_zero =
                haskey(f, "phonons/is_near_zero") ?
                Int8.(read(f["phonons/is_near_zero"])) .!= 0 :
                nothing,

            units =
                haskey(f, "units") ?
                String(read(f["units"])) :
                "eV",
        )
    end
end


# -----------------------------------------------------------------------------
# Load U_mu,n(k)
# -----------------------------------------------------------------------------

function load_unitary_data(filename::AbstractString)
    isfile(filename) ||
        error("Unitary-transformation file does not exist: $filename")

    return h5open(filename, "r") do f
        required = [
            "number_of_primitive_orbitals",
            "number_of_k_points",
            "repetitions_n1_n2_n3",
            "grids/k_integer_indices",
            "grids/k_reduced_primitive_reciprocal",
            "bands/energy_eV",
            "bands/U_mu_n_k",
        ]

        for path in required
            require_dataset(f, path, filename)
        end

        repetitions_vector =
            Vector{Int}(read(f["repetitions_n1_n2_n3"]))

        length(repetitions_vector) == 3 ||
            error("Expected three supercell repetitions.")

        U =
            ComplexF64.(read(f["bands/U_mu_n_k"]))

        energies =
            Matrix{Float64}(read(f["bands/energy_eV"]))

        norb =
            Int(read(f["number_of_primitive_orbitals"]))
        nk =
            Int(read(f["number_of_k_points"]))

        size(U, 1) == norb ||
            error("U first axis does not match number of orbitals.")

        size(U, 2) == norb ||
            error(
                "This script expects the complete square orbital-to-band " *
                "transformation: number of bands = number of orbitals."
            )

        size(U, 3) == nk ||
            error("U k-axis does not match number_of_k_points.")

        size(energies) == (norb, nk) ||
            error(
                "Band-energy dimensions $(size(energies)) do not match " *
                "expected ($norb,$nk)."
            )

        orbital_metadata = if haskey(f, "orbitals/mu")
            (
                mu =
                    Vector{Int}(read(f["orbitals/mu"])),
                primitive_atom_alpha =
                    haskey(f, "orbitals/primitive_atom_alpha") ?
                    Vector{Int}(
                        read(
                            f[
                                "orbitals/primitive_atom_alpha"
                            ]
                        )
                    ) :
                    Int[],
                local_orbital =
                    haskey(f, "orbitals/local_orbital") ?
                    Vector{Int}(
                        read(f["orbitals/local_orbital"])
                    ) :
                    Int[],
                orbital_token =
                    haskey(f, "orbitals/orbital_token") ?
                    h5_strings(f["orbitals/orbital_token"]) :
                    String[],
                basis_label =
                    haskey(f, "orbitals/basis_label") ?
                    h5_strings(f["orbitals/basis_label"]) :
                    String[],
                spin_slot =
                    haskey(f, "orbitals/spin_slot") ?
                    Vector{Int}(read(f["orbitals/spin_slot"])) :
                    Int[],
            )
        else
            nothing
        end

        return (
            filename = String(filename),
            number_of_orbitals = norb,
            number_of_bands = norb,
            number_of_k_points = nk,
            repetitions = Tuple(repetitions_vector),

            k_integer =
                normalized_integer_grid(
                    read(f["grids/k_integer_indices"])
                ),
            k_reduced =
                Matrix{Float64}(
                    read(
                        f[
                            "grids/k_reduced_primitive_reciprocal"
                        ]
                    )
                ),

            energies_eV = energies,
            U = U,
            orbital_metadata = orbital_metadata,

            bloch_basis_definition =
                haskey(f, "bloch_basis_definition") ?
                String(read(f["bloch_basis_definition"])) :
                "",

            band_state_definition =
                haskey(f, "band_state_definition") ?
                String(read(f["band_state_definition"])) :
                "",
        )
    end
end


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

function validate_inputs(g, unitary)
    g.number_of_primitive_orbitals ==
        unitary.number_of_orbitals ||
        error(
            "Orbital-g and unitary files use different numbers of orbitals: " *
            "$(g.number_of_primitive_orbitals) vs " *
            "$(unitary.number_of_orbitals)."
        )

    g.number_of_k_points ==
        unitary.number_of_k_points ||
        error(
            "Orbital-g and unitary files use different numbers of k points."
        )

    g.repetitions == unitary.repetitions ||
        error(
            "Orbital-g and unitary files use different supercell repetitions: " *
            "$(g.repetitions) vs $(unitary.repetitions)."
        )

    g.k_integer == unitary.k_integer ||
        error(
            "The k-grid ordering differs between electron_phonon_g_kq.h5 " *
            "and unitary_transformation_matrices.h5."
        )

    isapprox(
        g.k_reduced,
        unitary.k_reduced;
        rtol = 0.0,
        atol = 1.0e-12,
    ) || error(
        "Reduced k coordinates differ between the two input files."
    )

    size(g.k_integer, 2) == g.number_of_k_points ||
        error("g-file k grid has wrong size.")

    size(g.q_integer, 2) == g.number_of_q_points ||
        error("g-file q grid has wrong size.")

    for mode in g.block_mode
        1 <= mode <= g.number_of_modes ||
            error("Invalid phonon mode index $mode in orbital-g file.")
    end

    for mu in g.block_mu
        1 <= mu <= g.number_of_primitive_orbitals ||
            error("Invalid orbital mu=$mu in orbital-g file.")
    end

    for nu in g.block_nu
        1 <= nu <= g.number_of_primitive_orbitals ||
            error("Invalid orbital nu=$nu in orbital-g file.")
    end

    if !isnothing(g.valid_for_g)
        size(g.valid_for_g) ==
            (g.number_of_modes, g.number_of_q_points) ||
            error(
                "phonons/valid_for_g has inconsistent dimensions."
            )
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Index orbital g blocks by mode
# -----------------------------------------------------------------------------

function group_orbital_blocks_by_mode(g)
    by_mode =
        [
            Int[]
            for _ in 1:g.number_of_modes
        ]

    seen =
        Set{Tuple{Int,Int,Int}}()

    for entry in 1:g.number_of_blocks
        mode = g.block_mode[entry]
        mu = g.block_mu[entry]
        nu = g.block_nu[entry]

        key = (mode, mu, nu)
        key in seen &&
            error(
                "Duplicate orbital-g block for " *
                "(mode,mu,nu)=($mode,$mu,$nu)."
            )

        push!(seen, key)
        push!(by_mode[mode], entry)
    end

    return by_mode
end


# -----------------------------------------------------------------------------
# k+q lookup
# -----------------------------------------------------------------------------

function build_k_plus_q_lookup(g)
    repetitions = g.repetitions
    nk = g.number_of_k_points
    nq = g.number_of_q_points

    k_lookup =
        Dict{CellVector,Int}()

    for ik in 1:nk
        k = triplet_column(g.k_integer, ik)

        haskey(k_lookup, k) &&
            error("Duplicate integer k-grid point: $k")

        k_lookup[k] = ik
    end

    kplusq =
        zeros(Int, nk, nq)

    for iq in 1:nq
        q = triplet_column(g.q_integer, iq)

        for ik in 1:nk
            k = triplet_column(g.k_integer, ik)
            K = add_cells(k, q, repetitions)

            haskey(k_lookup, K) ||
                error(
                    "k+q point $K is absent from the unitary k grid. " *
                    "The grids are not closed under addition."
                )

            kplusq[ik, iq] = k_lookup[K]
        end
    end

    return kplusq
end


# -----------------------------------------------------------------------------
# Output-file initialization
# -----------------------------------------------------------------------------

function initialize_output_file(
    filename::AbstractString,
    g,
    unitary,
)
    temporary = String(filename) * ".tmp"
    isfile(temporary) && rm(temporary; force=true)

    nk = g.number_of_k_points
    nq = g.number_of_q_points
    nmodes = g.number_of_modes
    nbands = unitary.number_of_bands

    h5open(temporary, "w") do f
        f["format_version"] = 1

        f["number_of_k_points"] = nk
        f["number_of_q_points"] = nq
        f["number_of_modes"] = nmodes
        f["number_of_bands"] = nbands
        f["number_of_primitive_orbitals"] =
            g.number_of_primitive_orbitals
        f["repetitions_n1_n2_n3"] =
            collect(g.repetitions)

        f["units"] = g.units
        h5open(ORBITAL_G_FILE, "r") do source
            for key in ("temperature_K", "snapshot_indices", "number_of_snapshots")
                haskey(source, key) && (f[key] = read(source[key]))
            end
        end

        f["formula"] =
            "g_band_s_mn(k+q,k) = sum_mu,nu conj(U_mu_m(k+q)) * g_orbital_s_munu(k+q,k) * U_nu_n(k)"

        f["matrix_form"] =
            "g_band_s(k+q,k) = U(k+q)' * g_orbital_s(k+q,k) * U(k)"

        f["g_matrix_axis_1"] = "k index"
        f["g_matrix_axis_2"] = "q index"

        sources = create_group(f, "sources")
        sources["orbital_basis_g"] =
            abspath(ORBITAL_G_FILE)
        sources["unitary_transformations"] =
            abspath(UNITARY_FILE)
        h5open(UNITARY_FILE,"r") do reference
            if haskey(reference,"electronic_reference_definition")
                f["electronic_reference_definition"]=read(reference["electronic_reference_definition"])
            end
        end

        conventions = create_group(f, "conventions")
        conventions["bloch_basis_definition"] =
            unitary.bloch_basis_definition
        conventions["band_state_definition"] =
            unitary.band_state_definition
        conventions["initial_state"] =
            "band n at k"
        conventions["final_state"] =
            "band m at k+q"
        conventions["band_ordering"] =
            "same band ordering as unitary_transformation_matrices.h5"

        grids = create_group(f, "grids")
        grids["k_integer_indices"] =
            g.k_integer
        grids["q_integer_indices"] =
            g.q_integer
        grids["k_reduced_primitive_reciprocal"] =
            g.k_reduced
        grids["q_reduced_primitive_reciprocal"] =
            g.q_reduced
        grids["k_cartesian_inv_angstrom"] =
            g.k_cartesian
        grids["q_cartesian_inv_angstrom"] =
            g.q_cartesian
        grids["g_matrix_axis_1"] =
            "k index"
        grids["g_matrix_axis_2"] =
            "q index"

        bands = create_group(f, "bands")
        bands["energy_eV"] =
            unitary.energies_eV
        bands["energy_axes"] =
            "band, k_index"

        # Keep orbital metadata as provenance for the basis that was transformed.
        if !isnothing(unitary.orbital_metadata)
            metadata = unitary.orbital_metadata
            orbitals = create_group(f, "source_orbitals")

            orbitals["mu"] = metadata.mu

            if !isempty(metadata.primitive_atom_alpha)
                orbitals["primitive_atom_alpha"] =
                    metadata.primitive_atom_alpha
            end
            if !isempty(metadata.local_orbital)
                orbitals["local_orbital"] =
                    metadata.local_orbital
            end
            if !isempty(metadata.orbital_token)
                orbitals["orbital_token"] =
                    metadata.orbital_token
            end
            if !isempty(metadata.basis_label)
                orbitals["basis_label"] =
                    metadata.basis_label
            end
            if !isempty(metadata.spin_slot)
                orbitals["spin_slot"] =
                    metadata.spin_slot
            end
        end

        # Copy the phonon quantities that identify the mode and validity.
        phonons = create_group(f, "phonons")

        if !isnothing(g.frequency_THz)
            phonons["frequency_THz_signed"] =
                g.frequency_THz
        end

        if !isnothing(g.omega2)
            phonons["omega_squared_s_minus2"] =
                g.omega2
        end

        if !isnothing(g.is_imaginary)
            phonons["is_imaginary"] =
                Int8.(g.is_imaginary)
        end

        if !isnothing(g.is_near_zero)
            phonons["is_near_zero"] =
                Int8.(g.is_near_zero)
        end

        if !isnothing(g.valid_for_g)
            phonons["valid_for_g"] =
                Int8.(g.valid_for_g)
        end

        # -------------------------------------------------------------
        # Pre-create one Nk x Nq output dataset for every
        # (mode, final band m, initial band n).
        #
        # Chunking by (Nk,1) is deliberate because the calculation writes
        # one q-column at a time.
        # -------------------------------------------------------------

        g_blocks = create_group(f, "g_blocks")

        block_mode = Int[]
        block_m = Int[]
        block_n = Int[]
        block_dataset = String[]

        block_counter = 0

        for mode in 1:nmodes
            for m in 1:nbands
                for n in 1:nbands
                    block_counter += 1

                    dataset_name =
                        "block_" *
                        lpad(string(block_counter), 8, '0')

                    dataset =
                        create_dataset(
                            g_blocks,
                            dataset_name,
                            ComplexF64,
                            (nk, nq);
                            chunk = (nk, 1),
                        )
                    close(dataset)

                    push!(block_mode, mode)
                    push!(block_m, m)
                    push!(block_n, n)
                    push!(block_dataset, dataset_name)
                end
            end
        end

        index = create_group(f, "block_index")
        index["mode"] = block_mode

        # Descriptive names.
        index["band_m"] = block_m
        index["band_n"] = block_n

        # Short aliases convenient for later scripts.
        index["m"] = block_m
        index["n"] = block_n

        index["dataset_name"] =
            block_dataset
        index["number_of_blocks"] =
            block_counter
        index["number_of_bands"] =
            nbands
    end

    return temporary
end


# -----------------------------------------------------------------------------
# Streaming orbital -> band transformation
# -----------------------------------------------------------------------------

function transform_and_write!(temporary_output::AbstractString,g,unitary,blocks_by_mode,kplusq)
    nk=g.number_of_k_points
    nq=g.number_of_q_points
    nmodes=g.number_of_modes
    norb=g.number_of_primitive_orbitals
    nbands=unitary.number_of_bands
    qblock=min(nq,parse(Int,get(ENV,"EPH_Q_BLOCK","16")))
    qblock>0 || error("EPH_Q_BLOCK must be positive.")
    nan_complex=ComplexF64(NaN,NaN)
    Gorb=zeros(ComplexF64,norb,norb,nk,qblock)
    Gband=zeros(ComplexF64,nbands,nbands,nk,qblock)
    intermediate=zeros(ComplexF64,nbands,norb)
    output_name(mode,m,n)="block_"*lpad(string((mode-1)*nbands*nbands+(m-1)*nbands+n),8,'0')
    h5open(ORBITAL_G_FILE,"r") do fg
        h5open(temporary_output,"r+") do fo
            for mode in 1:nmodes
                entries=blocks_by_mode[mode]
                input_datasets=[fg["g_blocks/"*g.block_dataset[e]] for e in entries]
                output_datasets=[fo["g_blocks/"*output_name(mode,m,n)] for m in 1:nbands,n in 1:nbands]
                try
                    for first_q in 1:qblock:nq
                        qr=first_q:min(first_q+qblock-1,nq)
                        count=length(qr)
                        fill!(Gorb,0)
                        for (entry,dataset) in zip(entries,input_datasets)
                            values=ComplexF64.(dataset[:,qr])
                            size(values)==(nk,count) || error("Unexpected orbital-g q-block dimensions.")
                            @views Gorb[g.block_mu[entry],g.block_nu[entry],:,1:count].=values
                        end
                        for (local_q,iq) in enumerate(qr)
                            valid=isnothing(g.valid_for_g) || g.valid_for_g[mode,iq]
                            if !valid
                                @views fill!(Gband[:,:,:,local_q],nan_complex)
                                continue
                            end
                            for ik in 1:nk
                                iK=kplusq[ik,iq]
                                Uk=@view unitary.U[:,:,ik]
                                UK=@view unitary.U[:,:,iK]
                                # Same U(k+q)' * g(k,q) * U(k); reuse small buffers.
                                mul!(intermediate,adjoint(UK),@view(Gorb[:,:,ik,local_q]))
                                mul!(@view(Gband[:,:,ik,local_q]),intermediate,Uk)
                            end
                        end
                        for m in 1:nbands,n in 1:nbands
                            output_datasets[m,n][:,qr]=Array(@view Gband[m,n,:,1:count])
                        end
                        println("Mode $mode/$nmodes: q $(last(qr))/$nq written")
                        flush(stdout)
                        GC.gc(false)
                    end
                finally
                    foreach(close,input_datasets)
                    foreach(close,output_datasets)
                end
            end
        end
    end
    return nothing
end


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    for filename in (
        ORBITAL_G_FILE,
        UNITARY_FILE,
    )
        isfile(filename) ||
            error(
                "Required input file does not exist: $filename"
            )
    end

    println("Orbital-basis g: ", ORBITAL_G_FILE)
    println("Unitary matrices: ", UNITARY_FILE)
    println("Output:           ", OUTPUT_FILE)

    println("Loading metadata and U(k)...")
    g =
        load_orbital_g_metadata(
            ORBITAL_G_FILE
        )
    unitary =
        load_unitary_data(
            UNITARY_FILE
        )

    validate_inputs(g, unitary)

    println(
        "Norb = Nbands = ",
        unitary.number_of_bands,
    )
    println(
        "Nk = ",
        g.number_of_k_points,
        ", Nq = ",
        g.number_of_q_points,
        ", Nmodes = ",
        g.number_of_modes,
    )

    blocks_by_mode =
        group_orbital_blocks_by_mode(g)

    kplusq =
        build_k_plus_q_lookup(g)

    println("Creating output HDF5 structure...")
    temporary =
        initialize_output_file(
            OUTPUT_FILE,
            g,
            unitary,
        )

    println(
        "Transforming g_orbital -> g_band with " *
        "U(k+q)' * g * U(k)..."
    )

    try
        transform_and_write!(
            temporary,
            g,
            unitary,
            blocks_by_mode,
            kplusq,
        )

        mv(
            temporary,
            OUTPUT_FILE;
            force = true,
        )
    catch
        # Keep the temporary file for debugging rather than replacing any
        # previously successful final result.
        println(
            "Transformation failed. Partial temporary file retained at: ",
            temporary,
        )
        rethrow()
    end

    println(
        "Saved band-basis electron-phonon couplings: ",
        OUTPUT_FILE,
    )
    println("Done.")

    return nothing
end

main()
