module DirectFourierEph

using HDF5, LinearAlgebra, SparseArrays, Printf, FFTW, SHA
include(joinpath(@__DIR__, "DirectFourierKernel.jl"))
using .DirectFourierKernel
include(joinpath(@__DIR__, "ElectronPhononCouplingTools.jl"))
include(joinpath(@__DIR__, "HamsterHrLoader.jl"))
include(joinpath(@__DIR__, "PrimitiveBasisRelabel.jl"))
const EP = ElectronPhononCouplingTools
const VT = EP.VaspTrajectoryStatistics
const HR = HamsterHrLoader
const PB = PrimitiveBasisRelabel
export run_direct_eph, accumulate_fourier_snapshot!
include(joinpath(@__DIR__, "DirectDConvergence.jl"))

# C is fixed, with exactly the eigencutoff in ElectronPhononCouplingTools.
# z(t) = C^+ [u(t)-mean(u)_H]. For a=(alpha,Cartesian component),
# Z_a(q,t) = sum_L exp(+2pi*i*q.L) z_(a,L)(t).
# T_munu(K,k,t) = sum_r,r' exp(-2pi*i*K.r) H_(mu,r),(nu,r')(t)
#                                               exp(+2pi*i*k.r').
# d_a,mn(k,q) = mean_t [Z_a(q,t) T_mn(k+q,k,t)] / (Nc*sqrt(Nc)).
# This is exactly Fourier[translation_average(C^+ B_H)], not an approximation
# making C translation invariant. No conjugation is applied to the uH product.

const CV = NTuple{3,Int}
const ALGORITHM_VERSION = "direct-fourier-fixed-mean-pair-only-v4"
const PREVIOUS_V3_DRIVER_SHA256 = "afcecb55b631ad72781abed405072b93eb744697a774e7ccd8e0faac0cb989c6"
const PREVIOUS_V3_DIAGNOSTICS_SHA256 = "9b7f7fa11e59fe4115b8b4ddf610447f812a646b11233ff2a35029a1753f0d8b"
cellindex(c, reps) = 1 + c[1] + reps[1]*(c[2] + reps[2]*c[3])
framekey(n) = "frame_" * lpad(string(n), 8, '0')
pairkey(mu, nu) = "pair_" * lpad(string(mu), 4, '0') * "_" * lpad(string(nu), 4, '0')
shardpath(work, mu, nu) = joinpath(work, pairkey(mu, nu) * ".h5")

# POSIX rename replaces the destination without first unlinking it. Both paths
# are on the same filesystem. Only one pair-sized replacement exists at a time.
function replace_closed_file(tmp, path)
    open(tmp,"r") do io
        ccall(:fsync,Cint,(Cint,),Base.fd(io)) == 0 ||
            error("Cannot synchronize candidate file $tmp; old checkpoint is unchanged.")
    end
    result = ccall(:rename,Cint,(Cstring,Cstring),tmp,path)
    result == 0 || error("Cannot rename $tmp to $path; errno=$(Base.Libc.errno())")
    return nothing
end

function atomic_h5(writer::Function, path)
    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    h5open(writer, tmp, "w")
    replace_closed_file(tmp,path)
    return nothing
end

function content_digest(path)
    isfile(path) || error("Required input does not exist: $path")
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

struct TrajectoryIndex
    header
    offsets::Vector{Int64}
    direct::BitVector
end

function index_trajectory(path)
    return open(path, "r") do io
        header = VT.read_vasp_header(io)
        offsets = Int64[]
        direct = BitVector()
        while !eof(io)
            offset = position(io)
            line = lowercase(strip(readline(io)))
            isempty(line) && continue
            isdirect = startswith(line, "direct")
            iscart = startswith(line, "cartesian") || startswith(line, "kpoints")
            (isdirect || iscart) && occursin("configuration", line) ||
                error("Unexpected XDATCAR marker: $line")
            push!(offsets, offset)
            push!(direct, isdirect)
            for _ in header.atom_species
                eof(io) && error("XDATCAR ends inside a frame.")
                readline(io)
            end
        end
        isempty(offsets) && error("No XDATCAR frames.")
        TrajectoryIndex(header, offsets, direct)
    end
end

function foreach_frame(callback, path, index, ids)
    natoms = length(index.header.atom_species)
    positions = Matrix{Float64}(undef, 3, natoms)
    open(path, "r") do io
        for id in ids
            seek(io, index.offsets[id])
            readline(io)
            for atom in 1:natoms
                positions[:, atom] .= VT.parse_float_triplet(readline(io))
            end
            fractional = index.direct[id] ? positions :
                index.header.lattice \ (index.header.coordinate_scale .* positions)
            callback(id, fractional)
        end
    end
end

function validate_geometry(index, equilibrium)
    index.header.atom_species == equilibrium.header.atom_species ||
        error("XDATCAR and equilibrium atom ordering/species differ.")
    isapprox(index.header.lattice, equilibrium.header.lattice;
             rtol=1e-8, atol=1e-8) || error("XDATCAR and equilibrium lattices differ.")
end

# Rewrite only the temporary hconf; all other relative paths keep the template's
# directory as working directory. Missing overridden tags are inserted.
function write_chunk_hconf(template, path, hamfile, xdatcar, ids)
    settings = Dict(
        "options" => ["ham_file" => abspath(hamfile),
                      "write_hr" => "true", "write_hk" => "false"],
        "supercell" => ["xdatcar" => abspath(xdatcar),
                        "Nconf" => string(length(ids)),
                        "Nconf_min" => string(first(ids)),
                        "Nconf_max" => string(last(ids)),
                        "config_inds" => join(ids, " ")],
    )
    seen = Set{String}()
    found = Set{String}()
    block = ""
    open(path, "w") do io
        for line in readlines(template)
            stripped = strip(line)
            beginmatch = match(r"^begin\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"i, stripped)
            if beginmatch !== nothing
                block = lowercase(beginmatch.captures[1])
                empty!(seen)
                push!(found, block)
                println(io, line)
                continue
            end
            if lowercase(stripped) == "end"
                if haskey(settings, block)
                    for (tag, value) in settings[block]
                        lowercase(tag) in seen || println(io, "    $tag = $value")
                    end
                end
                println(io, line)
                block = ""
                continue
            end
            tagmatch = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            replaced = false
            if haskey(settings, block) && tagmatch !== nothing
                tag = lowercase(tagmatch.captures[1])
                for (wanted, value) in settings[block]
                    if tag == lowercase(wanted)
                        tag in seen && error("Duplicate $tag in $template")
                        println(io, "    $wanted = $value")
                        push!(seen, tag)
                        replaced = true
                        break
                    end
                end
            end
            replaced || println(io, line)
        end
    end
    all(x -> x in found, ("options", "supercell")) ||
        error("hconf must contain Options and Supercell blocks.")
end

# Each process gets unique paths. A surviving old child cannot overwrite a
# new process's output file. Configs remain beside the template for relative paths.
function hamster_attempt(config, ids)
    attempt = mktempdir(config.scratch_dir;prefix="hamster_",cleanup=false)
    hamfile = joinpath(attempt,"ham_hr.h5")
    conf = joinpath(dirname(config.hconf_file),
                    ".hconf_pair_only_" * basename(config.scratch_dir) * "_" * basename(attempt))
    write_chunk_hconf(config.hconf_file,conf,hamfile,config.xdatcar_file,ids)
    println("HAMSTER requested frames: ",join(ids,", "))
    println("HAMSTER output: ",hamfile); flush(stdout)
    try
        cd(dirname(config.hconf_file)) do
            run(Cmd([config.hamster_executable,"--conf",conf]))
        end
    finally
        rm(conf;force=true)
    end
    return hamfile
end

function selected_hamster_groups(config, path)
    groups = filter(g -> g.snapshot_index > 0,HR.available_hamster_snapshots(path))
    if config.system !== nothing
        filter!(g -> g.system == String(config.system),groups)
    else
        length(unique(g.system for g in groups)) <= 1 ||
            error("Several HAMSTER systems found in $path; set SYSTEM.")
    end
    return groups
end

# Return the source file for each requested frame. No group is relabelled.
# A partial batch is not accumulated. Retain valid returned frames and obtain
# missing ones individually, so an incomplete batch need not block a restart.
function run_hamster(config, ids)
    hamfile = hamster_attempt(config,ids)
    groups = selected_hamster_groups(config,hamfile)
    actual = sort([g.snapshot_index for g in groups])
    length(unique(actual)) == length(actual) || error("Duplicate HAMSTER labels: $actual")
    all(id -> id in ids,actual) || error(
        "HAMSTER returned unexpected frames. Requested $(collect(ids)); found $actual. File: $hamfile")
    sources = Dict{Int,String}(id => hamfile for id in actual)
    missing = [id for id in ids if !haskey(sources,id)]
    if !isempty(missing)
        println("HAMSTER batch output incomplete. Requested: ",join(ids,", "),
                "; found: ",join(actual,", "),"; retrying missing frames individually: ",join(missing,", "))
        flush(stdout)
        length(ids) > 1 || error("Single-frame HAMSTER output is missing frame $(only(ids)).")
        for id in missing
            onefile = hamster_attempt(config,[id])
            onegroups = selected_hamster_groups(config,onefile)
            found = [g.snapshot_index for g in onegroups]
            found == [id] || error(
                "Single-frame retry failed: requested $id; found $found. File: $onefile")
            sources[id] = onefile
        end
    end
    length(sources) == length(ids) || error("Incomplete HAMSTER source map.")
    return sources
end

function remove_hamster_outputs(sources)
    for path in unique(collect(values(sources)))
        rm(dirname(path);recursive=true,force=true)
    end
    return nothing
end

function make_mapping(snapshot, config, covariance, equilibrium)
    mapping = PB.build_primitive_basis_map(
        config.primitive_file, config.equilibrium_file, config.hconf_file,
        snapshot.basis, snapshot.dim;
        lattice_tolerance=1e-7, position_tolerance_angstrom=1e-3)
    reps = Tuple(mapping.repetitions)
    reps == covariance.repetitions || error("Covariance repetitions differ.")
    natoms = length(equilibrium.header.atom_species)
    ndof = 3 * natoms
    length(covariance.dof_alpha) == ndof || error("Covariance DOF count differs.")
    nalpha = maximum(mapping.atom_to_primitive_atom)
    nc = prod(reps)
    nalpha * nc == natoms || error("Incomplete primitive atom map.")
    ionic_dof = zeros(Int, 3*nalpha, nc)
    for atom in 1:natoms, component in 1:3
        d = 3*(atom-1) + component
        alpha = mapping.atom_to_primitive_atom[atom]
        L = Tuple(mapping.atom_to_cell[atom])
        covariance.dof_alpha[d] == alpha || error("Covariance alpha order differs.")
        Tuple(covariance.dof_L[:,d]) == L || error("Covariance cell order differs.")
        VT.component_index(covariance.dof_component[d]) == component ||
            error("Covariance Cartesian order differs.")
        l = cellindex(L, reps)
        a = 3*(alpha-1) + component
        ionic_dof[a,l] == 0 || error("Duplicate mapped displacement DOF.")
        ionic_dof[a,l] = d
    end
    all(ionic_dof .> 0) || error("Missing displacement DOF.")
    mu = zeros(Int, snapshot.dim)
    R = zeros(Int, snapshot.dim)
    for i in 1:snapshot.dim
        address = PB.relabel_basis_index(mapping, i)
        mu[i] = address.primitive_orbital
        R[i] = cellindex(Tuple(address.cell), reps)
    end
    norb = maximum(mu)
    snapshot.dim == norb*nc || error("Incomplete electronic basis.")
    length(Set(zip(mu,R))) == snapshot.dim || error("Duplicate electronic address.")
    return (; reps, nc, nalpha, norb, ionic_dof, mu, R,
            dim=snapshot.dim, basis=snapshot.basis)
end

# Mapping metadata lives inside pair_0001_0001.h5, never a separate checkpoint.
function write_mapping(parent, mapping)
    f = create_group(parent,"mapping")
    for key in (:ionic_dof,:mu,:R,:basis)
        f[string(key)] = getproperty(mapping,key)
    end
    f["reps"] = collect(mapping.reps)
    f["nalpha"] = mapping.nalpha
    f["norb"] = mapping.norb
    f["dim"] = mapping.dim
end

function read_mapping(f)
    reps = Tuple(Int.(read(f["reps"])))
    return (;reps,nc=prod(reps),nalpha=Int(read(f["nalpha"])),
        norb=Int(read(f["norb"])),dim=Int(read(f["dim"])),
        ionic_dof=Int.(read(f["ionic_dof"])),mu=Int.(read(f["mu"])),
        R=Int.(read(f["R"])),basis=read(f["basis"]))
end

function validate_pair_headers(work_dir,signature,nwindows,window_size,total,ids)
    for name in readdir(work_dir)
        match(r"^pair_[0-9]{4}_[0-9]{4}\.h5$",name) === nothing && continue
        path = joinpath(work_dir,name)
        h5open(path,"r") do f
            signature_matches(read(f["signature"]),signature) || error(
                "Checkpoint inputs/settings do not match: $path. Existing pair files were not changed.")
            w = Int(read(f["completed_window"]))
            1 <= w <= nwindows || error("Invalid completed window in $path")
            expected = min(w*window_size,total)
            haskey(f,"number_of_snapshots") && Int(read(f["number_of_snapshots"])) != expected &&
                error("Snapshot count disagrees with completed window in $path")
            haskey(f,"last_frame") && Int(read(f["last_frame"])) != ids[expected] &&
                error("Last frame disagrees with snapshot count in $path")
            haskey(f,"sum_d") || error("Missing tensor sum in $path")
        end
    end
end

# Called only after existing pair signatures have been checked. Legacy auxiliary
# files are not read for restart; any scratch state is rebuilt from the inputs.
function discard_stale_temporaries(work_dir)
    old_names = Set(["basis_mapping.h5","basis_mapping.h5.tmp","run_identity.txt",
        "run_identity.txt.tmp","window_sparse.h5","window_sparse.h5.tmp",
        ".ham_hr_direct_chunk.h5",".d_monitor_tail.h5"])
    for name in readdir(work_dir)
        path = joinpath(work_dir,name)
        if name in old_names || occursin(r"^pair_[0-9]{4}_[0-9]{4}\.h5\.tmp$",name)
            isfile(path) && rm(path)
        elseif startswith(name,".direct_eph_session_")
            isdir(path) && rm(path;recursive=true,force=true)
        end
    end
    return nothing
end

function consume_stop_request(work_dir)
    path = joinpath(work_dir,"STOP_AFTER_WINDOW")
    if isfile(path)
        rm(path)
        println("SAFE TO STOP: completed pair checkpoints retained; no partial window is being started.")
        flush(stdout)
        return true
    end
    return false
end

function make_zhat(u, mean_u, eig, mapping)
    # Use the Hamiltonian subset's mean, NOT covariance.mean_u.
    centered = u .- mean_u
    coefficients = eig.eigenvectors' * centered
    coefficients .*= eig.inverse_eigenvalues
    z = eig.eigenvectors * coefficients
    ni = 3*mapping.nalpha
    Z = zeros(ComplexF64, ni, mapping.nc)
    grid = zeros(ComplexF64, mapping.reps...)
    for a in 1:ni
        for l in 1:mapping.nc
            grid[l] = z[mapping.ionic_dof[a,l]]
        end
        bfft!(grid)  # unnormalized +i Fourier phase
        Z[a,:] .= vec(grid)
    end
    return Z
end

# Cache only sparse electronic entries, never u_a H_ij products.
function cache_snapshot(group, H, Z, mapping, tolerance)
    npairs = mapping.norb^2
    indices = [Int[] for _ in 1:npairs]
    values = [ComplexF64[] for _ in 1:npairs]
    for j in 1:size(H,2), pointer in nzrange(H,j)
        value = H.nzval[pointer]
        abs(value) > tolerance || continue
        i = H.rowval[pointer]
        pair = mapping.mu[i] + mapping.norb*(mapping.mu[j]-1)
        linear = mapping.R[i] + mapping.nc*(mapping.R[j]-1)
        push!(indices[pair], linear)
        push!(values[pair], value)
    end
    group["Z"] = Z
    for nu in 1:mapping.norb, mu in 1:mapping.norb
        pair = mu + mapping.norb*(nu-1)
        isempty(indices[pair]) && continue
        p = create_group(group, pairkey(mu,nu))
        p["indices"] = indices[pair]
        p["values"] = values[pair]
    end
end


function shard_progress(path, signature)
    isfile(path) || return 0
    return h5open(path,"r") do f
        signature_matches(read(f["signature"]),signature) || error("Stale state: $path")
        Int(read(f["completed_window"]))
    end
end

function reduce_window(cache, window, ids, mapping, config, signature)
    kp = plus_indices(mapping.reps)
    nc = mapping.nc; ni = 3*mapping.nalpha
    H_RR = zeros(ComplexF64,nc,nc)
    T = similar(H_RR)
    grid = reshape(H_RR,mapping.reps...,mapping.reps...)
    pf = plan_fft!(grid,(1,2,3); flags=FFTW.ESTIMATE)
    pb = plan_bfft!(grid,(4,5,6); flags=FFTW.ESTIMATE)
    # Only one orbital pair's cumulative d product sum is resident.
    for mu in 1:mapping.norb, nu in 1:mapping.norb
        path = shardpath(config.work_dir,mu,nu)
        done = shard_progress(path,signature)
        done >= window && continue
        done == window-1 || error("Missing earlier window for ($mu,$nu).")
        S, history, N = if isfile(path)
            h5open(path,"r") do f
                (read(f["sum_d"]),
                 haskey(f,"convergence_history") ? Matrix{Float64}(read(f["convergence_history"])) : zeros(Float64,0,8),
                 checkpoint_count(f,config))
            end
        else
            (zeros(ComplexF64,nc,nc,ni),zeros(Float64,0,8),0)
        end
        eltype(S) == ComplexF64 || error("Pair sum must contain ComplexF64 values.")
        size(S) == (nc,nc,ni) || error("Wrong checkpoint tensor dimensions.")
        N == (window-1)*config.accumulation_window || error("Unexpected checkpoint count.")
        rows = Vector{Vector{Float64}}()
        tailpath = joinpath(config.scratch_dir,"d_monitor_tail.h5")
        h5open(cache,"r") do f
            signature_matches(read(f["signature"]),signature) || error("Stale window cache.")
            Int.(read(f["snapshot_indices"])) == ids || error("Wrong cache frames.")
            for first in 1:config.hamster_chunk_size:length(ids)
                chunkids = ids[first:min(first+config.hamster_chunk_size-1,length(ids))]
                M = min(config.convergence_m,length(chunkids))
                tail_Z = zeros(ComplexF64,ni,nc,M)
                tail_file = M>1 ? h5open(tailpath,"w") : nothing
                try
                    for (position,id) in enumerate(chunkids)
                        group = f[framekey(id)]
                        Z = ComplexF64.(read(group["Z"]))
                        key = pairkey(mu,nu)
                        if haskey(group,key)
                            fill!(H_RR,0)
                            pg = group[key]
                            indices = Int.(read(pg["indices"]))
                            values = ComplexF64.(read(pg["values"]))
                            for j in eachindex(indices)
                                H_RR[indices[j]] += values[j]
                            end
                            transform_pair!(T,H_RR,mapping.reps,kp,pf,pb)
                        else
                            fill!(T,0)
                        end
                        add_pair_product!(S,T,Z,nc)
                        N += 1
                        tail_index = position-(length(chunkids)-M)
                        if tail_index > 0
                            tail_Z[:,:,tail_index] .= Z
                            M>1 && (tail_file["T_"*string(tail_index)] = T)
                        end
                    end
                    norms = pair_diagnostics(S,tail_Z,T,tail_file,N,M;
                                diagnostic_q_block=config.diagnostic_q_block)
                    push!(rows,Float64[N,last(chunkids),norms[1],norms[2],norms[3],norms[4],M,N-M])
                finally
                    tail_file === nothing || close(tail_file)
                    isfile(tailpath) && rm(tailpath)
                end
            end
        end
        added = reduce(vcat,[reshape(row,1,:) for row in rows])
        history = vcat(history,added)
        atomic_h5(path) do f
            f["signature"] = signature
            f["completed_window"] = window
            f["number_of_snapshots"] = N
            f["last_frame"] = last(ids)
            f["primitive_orbital_mu"] = mu
            f["primitive_orbital_nu"] = nu
            f["estimator"] = "S_N/N; fixed full-selected-trajectory mean and fixed C_MD^+"
            mu == 1 && nu == 1 && write_mapping(f,mapping)
            f["sum_d"] = S
            f["convergence_history"] = history
        end
        S = nothing; GC.gc(false)
        println("Window $window: saved orbital pair ($mu,$nu), N=$N"); flush(stdout)
    end
    rebuild_d_log(config,mapping,signature)
end

function final_output(config, mapping, covariance, ids, signature, nwindows, mean_u)
    cells = VT.enumerate_cells(mapping.reps)
    nc = mapping.nc
    ni = 3*mapping.nalpha
    n = length(ids)
    eig = config.eig
    atomic_h5(config.output_file) do f
        f["format_version"] = 2
        f["direct_algorithm"] = ALGORITHM_VERSION
        f["signature"] = signature
        f["number_of_snapshots"] = n
        f["snapshot_indices"] = ids
        f["number_of_primitive_cells"] = nc
        f["repetitions_n1_n2_n3"] = collect(mapping.reps)
        f["number_of_primitive_atoms"] = mapping.nalpha
        f["number_of_primitive_orbitals"] = mapping.norb
        f["normalization"] = "1/sqrt(Nc)"
        f["phase_convention"] = "exp[-2pi*i*(k+q).R] exp[+2pi*i*k.Rprime]"
        sample = create_group(f,"sampling")
        sample["hamiltonian_number_of_snapshots"] = n
        sample["hamiltonian_snapshot_indices"] = ids
        sample["covariance_number_of_snapshots"] = covariance.number_of_snapshots
        sample["covariance_snapshot_indices"] = covariance.snapshot_indices
        sample["mean_u_hamiltonian_subset_angstrom"] = mean_u
        sample["estimator"] = "Fourier[translation_average(C_MD^+ B_H)]"
        sample["hamiltonian_entry_tolerance_eV"] = config.h_tolerance
        modes = create_group(f,"covariance_pseudoinverse")
        modes["eigenvalues_angstrom2"] = eig.eigenvalues
        modes["retained_modes"] = Int.(eig.retained)
        modes["threshold_angstrom2"] = eig.threshold
        modes["rank"] = eig.rank
        sources = create_group(f,"sources")
        sources["xdatcar"] = config.xdatcar_file
        sources["displacement_covariance"] = config.covariance_file
        sources["hamster_config_template"] = config.hconf_file
        sources["equilibrium_poscar"] = config.equilibrium_file
        sources["primitive_poscar"] = config.primitive_file
        grids = create_group(f,"grids")
        integers = hcat([collect(c) for c in cells]...)
        reduced = Float64.(integers) ./ reshape(collect(mapping.reps),3,1)
        reciprocal = 2pi .* inv(config.primitive_lattice)'
        grids["k_integer_indices"] = integers
        grids["q_integer_indices"] = integers
        grids["k_reduced_primitive_reciprocal"] = reduced
        grids["q_reduced_primitive_reciprocal"] = reduced
        grids["primitive_reciprocal_lattice_inv_angstrom"] = reciprocal
        grids["k_cartesian_inv_angstrom"] = reciprocal*reduced
        grids["q_cartesian_inv_angstrom"] = reciprocal*reduced
        grids["d_matrix_axis_1"] = "k index"
        grids["d_matrix_axis_2"] = "q index"
        blocks = create_group(f,"d_blocks")
        alphas = Int[]; components = String[]; mus = Int[]; nus = Int[]
        datasets = String[]
        block_counter = 0
        # Match original block ordering: ionic row, then (mu,nu).
        # Read one ionic row per pair, not the whole final tensor.
        for a in 1:ni, mu in 1:mapping.norb, nu in 1:mapping.norb
            path = shardpath(config.work_dir,mu,nu)
            shard_progress(path,signature) == nwindows || error("Incomplete shard: $path")
            data = h5open(path,"r") do shard
                ComplexF64.(shard["sum_d"][:,:,a]) ./ n
            end
            block_counter += 1
            name = "block_" * lpad(string(block_counter),8,'0')
            blocks[name] = data
            push!(alphas,cld(a,3)); push!(components,VT.component_label(mod1(a,3)))
            push!(mus,mu); push!(nus,nu); push!(datasets,name)
        end
        bi = create_group(f,"block_index")
        bi["primitive_atom_alpha"] = alphas
        bi["component"] = components
        bi["mu"] = mus
        bi["nu"] = nus
        bi["dataset_name"] = datasets
        bi["number_of_blocks"] = block_counter
        f["is_complete"] = true
    end
end

function _run_direct_eph(;
    xdatcar_file, covariance_file, equilibrium_file, primitive_file, hconf_file,
    output_file, work_dir, model_files=String[], start_snapshot=500,
    final_snapshot=nothing, skip_snapshots=0, hamster_chunk_size=2,
    accumulation_window=50, h_tolerance=1e-14, pseudoinverse_rtol=1e-10,
    convergence_m=1, diagnostic_q_block=8,
    convergence_log_file=joinpath(dirname(hconf_file),"d_convergence.txt"),
    pseudoinverse_atol=0.0, system=nothing, fft_threads=1,
    hamster_executable="hamster", assemble_final_output=false, scratch_dir,
)
    start_snapshot >= 1 || error("START_SNAPSHOT must be positive.")
    skip_snapshots >= 0 || error("SKIP_SNAPSHOTS must be nonnegative.")
    hamster_chunk_size >= 1 && accumulation_window >= 1 || error("Invalid batch sizes.")
    h_tolerance >= 0 && pseudoinverse_rtol >= 0 && pseudoinverse_atol >= 0 ||
        error("Tolerances must be nonnegative.")
    1 <= convergence_m <= hamster_chunk_size || error("CONVERGENCE_M must be between 1 and HAMSTER_CHUNK_SIZE.")
    diagnostic_q_block >= 1 || error("DIAGNOSTIC_Q_BLOCK must be positive.")
    accumulation_window % hamster_chunk_size == 0 ||
        error("ACCUMULATION_WINDOW must be a multiple of HAMSTER_CHUNK_SIZE.")
    fft_threads >= 1 || error("FFT_THREADS must be positive.")
    FFTW.set_num_threads(fft_threads)
    xdatcar_file = abspath(xdatcar_file); covariance_file = abspath(covariance_file)
    equilibrium_file = abspath(equilibrium_file); primitive_file = abspath(primitive_file)
    hconf_file = abspath(hconf_file); output_file = abspath(output_file)
    work_dir = abspath(work_dir)
    convergence_log_file = abspath(convergence_log_file)
    mkpath(dirname(convergence_log_file))
    mkpath(work_dir); mkpath(dirname(output_file))
    println("Checking input identities for safe resume..."); flush(stdout)
    inputs = unique(vcat([xdatcar_file,covariance_file,equilibrium_file,primitive_file,
                          hconf_file, @__FILE__, joinpath(@__DIR__,"DirectFourierKernel.jl"),
                          joinpath(@__DIR__,"DirectDConvergence.jl"),
                          joinpath(@__DIR__,"ElectronPhononCouplingTools.jl"),
                          joinpath(@__DIR__,"VaspTrajectoryStatistics.jl"),
                          joinpath(@__DIR__,"HamsterHrLoader.jl"),
                          joinpath(@__DIR__,"PrimitiveBasisRelabel.jl")],abspath.(model_files)))
    identity = join([path * "=" * content_digest(path) for path in inputs], "\n")
    parameters = repr((ALGORITHM_VERSION,start_snapshot,final_snapshot,skip_snapshots,
                       accumulation_window,hamster_chunk_size,convergence_m,h_tolerance,pseudoinverse_rtol,
                       pseudoinverse_atol,system))
    signature = bytes2hex(sha256(identity * parameters))
    # Reconstruct only known previous implementations' identities from the same
    # physical inputs/settings. Never trust an old standalone manifest.
    source_path = abspath(@__FILE__)
    helper_path = joinpath(@__DIR__,"DirectDConvergence.jl")
    previous_identity = join([path * "=" *
        (path == source_path ? PREVIOUS_V3_DRIVER_SHA256 :
         path == helper_path ? PREVIOUS_V3_DIAGNOSTICS_SHA256 : content_digest(path))
        for path in inputs],"\n")
    previous_parameters = repr(("direct-fourier-fixed-mean-d-monitor-v3",start_snapshot,
        final_snapshot,skip_snapshots,accumulation_window,hamster_chunk_size,
        convergence_m,h_tolerance,pseudoinverse_rtol,pseudoinverse_atol,system))
    previous_signature = bytes2hex(sha256(previous_identity*previous_parameters))
    legacy_inputs = filter(path -> path != helper_path,inputs)
    legacy_identity = join([path * "=" *
        (path == source_path ? LEGACY_FIXED_MEAN_SOURCE_SHA256 : content_digest(path))
        for path in legacy_inputs],"\n")
    legacy_parameters = repr(("direct-fourier-fixed-C-v1",start_snapshot,final_snapshot,
        skip_snapshots,accumulation_window,h_tolerance,pseudoinverse_rtol,
        pseudoinverse_atol,system))
    legacy_signature = bytes2hex(sha256(legacy_identity*legacy_parameters))
    ACCEPTED_RUN_SIGNATURES[signature] = Set([signature,previous_signature,legacy_signature])
    equilibrium = VT.read_poscar(equilibrium_file)
    println("Indexing XDATCAR..."); flush(stdout)
    index = index_trajectory(xdatcar_file)
    validate_geometry(index,equilibrium)
    lastframe = isnothing(final_snapshot) ? length(index.offsets) : final_snapshot
    start_snapshot <= lastframe <= length(index.offsets) || error("Invalid snapshot range.")
    ids = collect(start_snapshot:(skip_snapshots+1):lastframe)
    println("Selected ",length(ids)," Hamiltonian frames; first=",first(ids),", last=",last(ids))
    println("Computing mean displacement of this exact subset (no Hamiltonians)...")
    mean_u = zeros(Float64,3*length(equilibrium.header.atom_species))
    foreach_frame(xdatcar_file,index,ids) do _, fractional
        mean_u .+= vec(VT.minimum_image_displacements(fractional,
                     equilibrium.fractional_positions,equilibrium.header.lattice))
    end
    mean_u ./= length(ids)
    covariance = EP.load_displacement_covariance(covariance_file)
    println("Preparing fixed covariance pseudoinverse..."); flush(stdout)
    eig = EP.covariance_eigensystem(covariance.covariance;
              relative_cutoff=pseudoinverse_rtol,absolute_cutoff=pseudoinverse_atol)
    println("Covariance retained rank: ",eig.rank)
    primitive_lattice = equilibrium.header.lattice *
                        Diagonal(1.0 ./ collect(covariance.repetitions))
    config = (; xdatcar_file,covariance_file,equilibrium_file,primitive_file,
        hconf_file,output_file,work_dir,hamster_executable,system,h_tolerance,
        eig,primitive_lattice,hamster_chunk_size,accumulation_window,scratch_dir,
        convergence_m,diagnostic_q_block,convergence_log_file,total_snapshots=length(ids))
    windows = [ids[i:min(i+accumulation_window-1,length(ids))]
               for i in 1:accumulation_window:length(ids)]
    validate_pair_headers(work_dir,signature,length(windows),accumulation_window,length(ids),ids)
    discard_stale_temporaries(work_dir)
    mkdir(scratch_dir)
    mapping = nothing
    anchor = shardpath(work_dir,1,1)
    if isfile(anchor)
        mapping = h5open(anchor,"r") do f
            haskey(f,"mapping") ? read_mapping(f["mapping"]) : nothing
        end
    end
    seed_sources = Dict{Int,String}()
    if mapping === nothing
        # Existing v3 pairs have no embedded mapping. Reconstruct it from a fresh
        # Hamiltonian, never from basis_mapping.h5. Reuse that same frame in the
        # pending cache window, rather than calculating it twice.
        counts = Int[]
        for name in readdir(work_dir)
            match(r"^pair_[0-9]{4}_[0-9]{4}\.h5$",name) === nothing && continue
            push!(counts,h5open(joinpath(work_dir,name),"r") do f
                checkpoint_count(f,config)
            end)
        end
        bootstrap_count = isempty(counts) ? 0 : minimum(counts)
        bootstrap_id = ids[min(bootstrap_count+1,length(ids))]
        println("Reconstructing basis mapping from fresh frame $bootstrap_id..."); flush(stdout)
        seed_sources = run_hamster(config,[bootstrap_id])
        snapshot = HR.load_hamster_snapshot(seed_sources[bootstrap_id],bootstrap_id;system=system)
        mapping = make_mapping(snapshot,config,covariance,equilibrium)
        snapshot = nothing; GC.gc(false)
    end
    mapping.reps == covariance.repetitions || error("Checkpoint mapping/covariance mismatch.")
    mapping.dim == mapping.norb*mapping.nc || error("Invalid embedded electronic mapping.")
    size(mapping.ionic_dof) == (3*mapping.nalpha,mapping.nc) || error("Invalid embedded ionic mapping.")
    pair_shape = (mapping.nc,mapping.nc,3*mapping.nalpha)
    for name in readdir(work_dir)
        parsed = match(r"^pair_([0-9]{4})_([0-9]{4})\.h5$",name)
        parsed === nothing && continue
        mu,nu = parse.(Int,parsed.captures)
        1 <= mu <= mapping.norb && 1 <= nu <= mapping.norb || error("Unexpected pair file: $name")
        h5open(joinpath(work_dir,name),"r") do f
            size(f["sum_d"]) == pair_shape || error("Wrong tensor dimensions in $name")
        end
    end
    progresses = [shard_progress(shardpath(work_dir,mu,nu),signature)
                  for mu in 1:mapping.norb for nu in 1:mapping.norb]
    maximum(progresses)-minimum(progresses) <= 1 || error(
        "Pair checkpoints span more than one incomplete window; no pair files were changed.")
    @printf("Full d pair-file payload: %.3f GiB; one pair replacement: %.3f MiB\n",
        16.0*prod(pair_shape)*mapping.norb^2/2.0^30,16.0*prod(pair_shape)/2.0^20)
    println("Pair checkpoint windows: minimum=",minimum(progresses),", maximum=",maximum(progresses))
    rebuild_d_log(config,mapping,signature)
    # At an existing common boundary the request may be honored immediately.
    if minimum(progresses) == maximum(progresses) && consume_stop_request(work_dir)
        remove_hamster_outputs(seed_sources)
        return nothing
    end
    for (window,windowids) in enumerate(windows)
        if all(shard_progress(shardpath(work_dir,mu,nu),signature) >= window
               for mu in 1:mapping.norb for nu in 1:mapping.norb)
            continue
        end
        # Always build afresh, even if an interrupted run left a complete cache.
        cache = joinpath(scratch_dir,"window_sparse.h5")
        println("Building fresh sparse cache window $window/$(length(windows))"); flush(stdout)
        h5open(cache,"w") do f
            f["signature"] = signature
            f["snapshot_indices"] = windowids
            for first in 1:hamster_chunk_size:length(windowids)
                chunkids = windowids[first:min(first+hamster_chunk_size-1,length(windowids))]
                cached_ids = [id for id in chunkids if haskey(seed_sources,id)]
                needed_ids = [id for id in chunkids if !haskey(seed_sources,id)]
                sources = isempty(needed_ids) ? Dict{Int,String}() : run_hamster(config,needed_ids)
                for id in cached_ids
                    sources[id] = seed_sources[id]
                end
                foreach_frame(xdatcar_file,index,chunkids) do id,fractional
                    snapshot = HR.load_hamster_snapshot(sources[id],id;system=system)
                    snapshot.dim == mapping.dim && isequal(snapshot.basis,mapping.basis) ||
                        error("HAMSTER basis changed at snapshot $id")
                    u = vec(VT.minimum_image_displacements(fractional,
                        equilibrium.fractional_positions,equilibrium.header.lattice))
                    Z = make_zhat(u,mean_u,eig,mapping)
                    group = create_group(f,framekey(id))
                    cache_snapshot(group,snapshot.H_supercell_pbc,Z,mapping,h_tolerance)
                end
                remove_hamster_outputs(sources)
                for id in cached_ids
                    delete!(seed_sources,id)
                end
            end
        end
        println("Reducing window $window; already committed pairs are skipped."); flush(stdout)
        reduce_window(cache,window,windowids,mapping,config,signature)
        rm(cache)
        println("WINDOW $window COMMITTED FOR ALL PAIRS; snapshots=",min(window*accumulation_window,length(ids)))
        flush(stdout)
        consume_stop_request(work_dir) && return nothing
    end
    remove_hamster_outputs(seed_sources)
    if assemble_final_output
        println("Explicitly requested: assembling an additional full d output file."); flush(stdout)
        final_output(config,mapping,covariance,ids,signature,length(windows),mean_u)
        println("Saved: $output_file")
    else
        println("COMPLETE: d is stored in the pair files (d_N = sum_d / number_of_snapshots).")
        println("No additional full-size d file was created.")
    end
    return nothing
end


# Lock is coordination only, not a progress checkpoint. Never unlink a live
# lock inode. Unique session paths also isolate surviving old HAMSTER children.
function run_direct_eph(;work_dir,hconf_file,kwargs...)
    mkpath(work_dir)
    open(joinpath(work_dir,".run.lock"),"a+") do io
        result = ccall(:flock,Cint,(Cint,Cint),Base.fd(io),6)
        result == 0 || error("Another driver is already using WORK_DIR: $work_dir")
        # The directory is created only after stale-session cleanup in _run.
        scratch_dir = joinpath(abspath(work_dir),".direct_eph_session_" * basename(tempname()))
        try
            _run_direct_eph(;work_dir,hconf_file,scratch_dir,kwargs...)
        finally
            # Keep failed scratch only until next invocation; it is never resumed.
            ccall(:flock,Cint,(Cint,Cint),Base.fd(io),8)
        end
    end
end

end # module
