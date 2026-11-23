# Included inside DirectFourierEph; uses its HDF5, FFTW and Printf imports.
# Small diagnostic slabs only. No second whole d block or whole B tensor.

function transform_pair!(T, H_RR, reps, kp, forward_plan, backward_plan)
    grid = reshape(H_RR,reps...,reps...)
    forward_plan * grid
    backward_plan * grid
    @inbounds for q in axes(T,2), k in axes(T,1)
        T[k,q] = H_RR[kp[k,q],k]
    end
    return T
end

# S = factor*sum_t Z_t(q) T_t(k,q), with Z built using the SAME
# precomputed mean displacement as the original direct driver.
# Its current prefix estimate is d_N=S_N/N. The earlier prefix is obtained by
# subtracting only its last M contributions. There is no change of centering.
function add_pair_product!(S, T, Z, nc)
    factor = inv(nc*sqrt(Float64(nc)))
    @inbounds for a in axes(Z,1), q in 1:nc
        weight = factor*Z[a,q]
        for k in 1:nc
            S[k,q,a] += weight*T[k,q]
        end
    end
    return nothing
end

# For M=1 use the existing last T in RAM. For M>1 the M large Fourier
# Hamiltonian matrices stay on disk and are read in small q-column slabs.
function pair_diagnostics(S, tail_Z, last_T, tail_file, N, M;
                          diagnostic_q_block=8)
    P = N-M
    nc = size(S,1)
    ni = size(S,3)
    factor = inv(nc*sqrt(Float64(nc)))
    old_norm2 = 0.0; new_norm2 = 0.0; difference_norm2 = 0.0; max_abs = 0.0
    for firstq in 1:diagnostic_q_block:nc
        qs = firstq:min(firstq+diagnostic_q_block-1,nc)
        nq = length(qs)
        delta_S = zeros(ComplexF64,nc,nq,ni)
        if P > 0
            for t in 1:M
                Tblock = M == 1 ? Matrix(@view last_T[:,qs]) :
                         ComplexF64.(tail_file["T_"*string(t)][:,qs])
                @inbounds for a in 1:ni, (j,q) in enumerate(qs)
                    weight = factor*tail_Z[a,q,t]
                    for k in 1:nc
                        delta_S[k,j,a] += weight*Tblock[k,j]
                    end
                end
            end
        end
        @inbounds for a in 1:ni, (j,q) in enumerate(qs), k in 1:nc
            new = S[k,q,a]/N
            new_norm2 += abs2(new)
            max_abs = max(max_abs,abs(new))
            if P > 0
                old = (S[k,q,a]-delta_S[k,j,a])/P
                old_norm2 += abs2(old)
                difference_norm2 += abs2(new-old)
            end
        end
    end
    return new_norm2,old_norm2,difference_norm2,max_abs
end

# Only the exact previously delivered fixed-mean v1 source is accepted for
# migration. Its input/sampling digest must still match the saved manifest.
const LEGACY_FIXED_MEAN_SOURCE_SHA256 = "b55bf7bc710da1d30c581e8726399a9795ee68195407d33dfdd48b49f9da20b5"
const ACCEPTED_RUN_SIGNATURES = Dict{String,Set{String}}()
signature_matches(stored, current) = stored in get(ACCEPTED_RUN_SIGNATURES,current,Set([current]))

function checkpoint_count(file, config)
    haskey(file,"number_of_snapshots") && return Int(read(file["number_of_snapshots"]))
    return min(Int(read(file["completed_window"]))*config.accumulation_window,
               config.total_snapshots)
end

# Pair history columns: N, last_frame, norm_new^2, norm_old^2, diff^2, maxabs, M, P.
function rebuild_d_log(config, mapping, signature)
    paths = [shardpath(config.work_dir,mu,nu)
             for mu in 1:mapping.norb for nu in 1:mapping.norb]
    safe_count = typemax(Int)
    for path in paths
        if !isfile(path)
            safe_count = 0
            break
        end
        safe_count = min(safe_count,h5open(path,"r") do f
            signature_matches(read(f["signature"]),signature) || error("Stale d checkpoint: $path")
            checkpoint_count(f,config)
        end)
    end
    safe_count == typemax(Int) && (safe_count=0)
    totals = Dict{Int,Vector{Float64}}()
    contributions = Dict{Int,Int}()
    if safe_count > 0
        for path in paths
            history = h5open(path,"r") do f
                haskey(f,"convergence_history") ? Matrix{Float64}(read(f["convergence_history"])) :
                    zeros(Float64,0,8)
            end
            for row in axes(history,1)
                N = round(Int,history[row,1])
                N <= safe_count || continue
                if !haskey(totals,N)
                    totals[N] = [history[row,2],0.0,0.0,0.0,0.0,
                                 history[row,7],history[row,8]]
                    contributions[N] = 0
                end
                aggregate = totals[N]
                aggregate[1] == history[row,2] && aggregate[6] == history[row,7] &&
                    aggregate[7] == history[row,8] || error("Convergence histories disagree.")
                aggregate[2] += history[row,3]
                aggregate[3] += history[row,4]
                aggregate[4] += history[row,5]
                aggregate[5] = max(aggregate[5],history[row,6])
                contributions[N] += 1
            end
        end
    end
    log = config.convergence_log_file
    # The text file is a derived report, never a checkpoint. Compatible old
    # signatures are accepted while pairs are migrated; progress comes from pairs.
    if isfile(log) && filesize(log) > 0
        firstline = open(readline,log)
        prefix = "# Direct d convergence; run_signature="
        if startswith(firstline,prefix)
            previous_signature = firstline[length(prefix)+1:end]
            signature_matches(previous_signature,signature) || error(
                "Convergence report belongs to another run: $log. Choose another log filename.")
        else
            error("Refusing to replace a non-direct convergence report: $log")
        end
    end
    tmp = log * ".tmp"
    previous_norm = nothing
    components = (3*mapping.nalpha)*mapping.norb^2*mapping.nc^2
    open(tmp,"w") do io
        println(io,"# Direct d convergence; run_signature="*signature)
        println(io,"# d_N = S_N/N with the SAME precomputed full-selected-trajectory displacement mean and fixed C_MD^+ at every N.")
        println(io,"# The prefix d_N is the current direct-driver estimator, not a covariance recentered separately on each prefix.")
        println(io,"# Rows correspond to HAMSTER chunk ends; rows become available after all orbital pairs finish their cache window.")
        println(io,"# r_cumulative = ||d_(P+M)-d_P||_F/max(||d_(P+M)||_F,||d_P||_F).")
        println(io,"# relative_change_d_norm compares norms at consecutive logged HAMSTER chunk ends.")
        println(io,"# Norm/RMS/max units: eV/angstrom. Relative columns dimensionless; NaN means no usable reference.")
        println(io,"# N_snapshots last_frame d_norm_eV_per_A d_rms_eV_per_A d_max_abs_eV_per_A relative_change_d_norm r_cumulative N_added N_previous")
        for N in sort!(collect(keys(totals)))
            # Legacy pairs may have finished a window before monitoring was enabled.
            # Such rows cannot be recovered from their sums alone; omit them.
            contributions[N] == length(paths) || continue
            row = totals[N]
            lastframe = round(Int,row[1]); M=round(Int,row[6]); P=round(Int,row[7])
            newnorm = sqrt(max(0.0,row[2])); oldnorm=sqrt(max(0.0,row[3]))
            denominator=max(newnorm,oldnorm)
            r = P>0 && denominator>0 ? sqrt(max(0.0,row[4]))/denominator : NaN
            relative = previous_norm === nothing ? NaN :
                (max(newnorm,previous_norm)>0 ?
                 abs(newnorm-previous_norm)/max(newnorm,previous_norm) : NaN)
            @printf(io,"%d  %d  %.12e  %.12e  %.12e  %.12e  %.12e  %d  %d\n",
                    N,lastframe,newnorm,newnorm/sqrt(components),row[5],relative,r,M,P)
            previous_norm = newnorm
        end
    end
    replace_closed_file(tmp,log)
    println("Convergence log synchronized through N=$safe_count: $log"); flush(stdout)
    return nothing
end
