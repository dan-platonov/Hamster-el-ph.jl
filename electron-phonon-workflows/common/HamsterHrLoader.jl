module HamsterHrLoader

using HDF5
using SparseArrays

export available_hamster_snapshots,
       load_hamster_snapshot,
       get_HR

const RVector = NTuple{3, Int}

"""Read one sparse CSC matrix from a HAMSTER HDF5 subgroup."""
function read_sparse_block(parent, block_name::AbstractString)
    haskey(parent, block_name) ||
        error("Sparse block \"$block_name\" does not exist.")

    g = parent[block_name]
    m = Int(read(g["m"]))
    n = Int(read(g["n"]))
    colptr = Vector{Int}(read(g["colptr"]))
    rowval = Vector{Int}(read(g["rowval"]))
    nzval = vec(read(g["nzval"]))

    length(colptr) == n + 1 ||
        error("Invalid CSC block $block_name: expected n+1 column pointers.")
    length(rowval) == length(nzval) ||
        error("Invalid CSC block $block_name: rowval and nzval lengths differ.")

    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

"""
Interpret HAMSTER real-space group names.

Examples:
- `Hr` -> static single-structure Hamiltonian, index 0
- `Hr__4000` -> trajectory index 4000, empty system label
- `Hr_GaAs_4000` -> trajectory index 4000, system `GaAs`
"""
function parse_Hr_group_name(group_name::AbstractString)
    name = String(group_name)

    if name == "Hr"
        return (groupname=name, system="", snapshot_index=0)
    end

    result = match(r"^Hr_(.*)_([0-9]+)$", name)
    isnothing(result) && return nothing

    return (
        groupname=name,
        system=result.captures[1],
        snapshot_index=parse(Int, result.captures[2]),
    )
end

"""Return all real-space Hamiltonian groups available in `filename`."""
function available_hamster_snapshots(filename::AbstractString)
    isfile(filename) || error("Hamiltonian file does not exist: $filename")

    return h5open(filename, "r") do f
        result = NamedTuple[]
        for key in keys(f)
            parsed = parse_Hr_group_name(String(key))
            isnothing(parsed) || push!(result, parsed)
        end
        sort!(result; by=x -> (x.system, x.snapshot_index))
        result
    end
end

function find_Hr_group(
    f,
    snapshot_index::Integer;
    system::Union{Nothing, AbstractString}=nothing,
)
    snapshot_index >= 0 ||
        throw(ArgumentError("snapshot_index must be non-negative."))

    candidates = NamedTuple[]
    for key in keys(f)
        parsed = parse_Hr_group_name(String(key))
        isnothing(parsed) && continue
        parsed.snapshot_index == snapshot_index || continue
        isnothing(system) || parsed.system == String(system) || continue
        push!(candidates, parsed)
    end

    if isempty(candidates)
        available = sort(
            String(key) for key in keys(f) if startswith(String(key), "Hr")
        )
        error(
            "No real-space Hamiltonian was found for snapshot " *
            "$snapshot_index. Available groups:\n" * join(available, "\n")
        )
    elseif length(candidates) > 1
        names = join((x.groupname for x in candidates), "\n")
        error(
            "Several systems contain snapshot $snapshot_index:\n$names\n" *
            "Pass system=\"...\" explicitly."
        )
    end

    return only(candidates)
end

function standardize_R_vectors(raw)
    ndims(raw) == 2 || error("The vecs dataset must be two-dimensional.")

    if size(raw, 1) == 3
        return Int.(raw)
    elseif size(raw, 2) == 3
        return permutedims(Int.(raw))
    end

    error("Unexpected vecs shape $(size(raw)); expected 3×NR or NR×3.")
end

function read_basis(f, system::AbstractString)
    expected = isempty(system) ? "basis" : "basis_$system"
    if haskey(f, expected)
        return read(f[expected]), expected
    end

    candidates = sort(
        String(key) for key in keys(f) if startswith(String(key), "basis")
    )

    if isempty(candidates)
        return nothing, nothing
    elseif length(candidates) == 1
        key = only(candidates)
        @warn "Using basis dataset $key instead of expected dataset $expected."
        return read(f[key]), key
    end

    error(
        "Could not select a basis dataset for system \"$system\". " *
        "Available datasets: $(join(candidates, ", "))."
    )
end

"""
    load_hamster_snapshot(filename, snapshot_index; system=nothing)

Load one HAMSTER trajectory Hamiltonian. The returned `H_supercell_pbc` is

    sum_Rsc H(Rsc),

namely the Hamiltonian acting on a wavefunction periodic over the MD
supercell. The individual blocks remain available in `Hr` for diagnostics.
"""
function load_hamster_snapshot(
    filename::AbstractString,
    snapshot_index::Integer;
    system::Union{Nothing, AbstractString}=nothing,
)
    isfile(filename) || error("Hamiltonian file does not exist: $filename")

    return h5open(filename, "r") do f
        selected = find_Hr_group(f, snapshot_index; system=system)
        groupname = selected.groupname
        system_name = selected.system
        g = f[groupname]

        haskey(g, "vecs") || error("Group $groupname has no vecs dataset.")
        R_vectors = standardize_R_vectors(read(g["vecs"]))
        nR = size(R_vectors, 2)
        nR > 0 || error("Group $groupname contains no R blocks.")

        first_block = ComplexF64.(read_sparse_block(g, "1"))
        m, n = size(first_block)
        m == n || error("Hamiltonian block is not square: $(size(first_block)).")

        Hr = Dict{RVector, SparseMatrixCSC{ComplexF64, Int}}()
        H_supercell_pbc = spzeros(ComplexF64, m, n)

        for block_index in 1:nR
            block_name = string(block_index)
            haskey(g, block_name) ||
                error("Missing block $block_name in group $groupname.")

            H_R = block_index == 1 ?
                  first_block : ComplexF64.(read_sparse_block(g, block_name))
            size(H_R) == (m, n) ||
                error("Block $block_name has inconsistent dimensions.")

            R = (
                R_vectors[1, block_index],
                R_vectors[2, block_index],
                R_vectors[3, block_index],
            )

            if haskey(Hr, R)
                @warn "Translation $R occurs more than once; adding its blocks."
                Hr[R] = Hr[R] + H_R
            else
                Hr[R] = H_R
            end

            H_supercell_pbc = H_supercell_pbc + H_R
        end

        dropzeros!(H_supercell_pbc)
        basis, basis_key = read_basis(f, system_name)

        return (
            snapshot_index=Int(snapshot_index),
            groupname=groupname,
            system=system_name,
            basis_key=basis_key,
            basis=basis,
            dim=m,
            R_vectors=R_vectors,
            Hr=Hr,
            H_supercell_pbc=H_supercell_pbc,
        )
    end
end

"""Return an individual H(R) block, or a correctly sized sparse zero block."""
function get_HR(snapshot, R::RVector)
    return get(
        snapshot.Hr,
        R,
        spzeros(ComplexF64, snapshot.dim, snapshot.dim),
    )
end

end # module HamsterHrLoader
