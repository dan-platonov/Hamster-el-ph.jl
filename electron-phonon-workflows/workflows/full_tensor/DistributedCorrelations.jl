# Electronic columns are owned by contiguous MPI rank ranges. Dictionaries are
# changed only outside threaded loops; threads update disjoint numeric vectors.
const LOCAL_ACCUMULATOR = Ref{Any}(nothing)
const CHECKPOINT_KEYS = Ref(Vector{Tuple{Int,Int}}())
const CHECKPOINT_COUNTS = Ref(Int[])
const CHECKPOINT_MEAN_H = Dict{Tuple{Int,Int},ComplexF64}()
owned_column(column,dim) = min(NRANKS-1, ((column-1)*NRANKS) ÷ dim) == RANK

function local_accumulate!(acc,H,u,frame,marker,tolerance)
    size(H)==(acc.dim,acc.dim) || error("Changed Hamiltonian dimensions")
    size(u)==(3,acc.natoms) || error("Changed displacement dimensions")
    frame in acc.snapshot_indices && error("Duplicate snapshot $frame")
    uv=vec(u)
    acc.sum_u .+= uv
    buffers=Vector{Vector{ComplexF64}}()
    values=ComplexF64[]
    for column in 1:acc.dim
        owned_column(column,acc.dim) || continue
        for p in H.colptr[column]:(H.colptr[column+1]-1)
            value=H.nzval[p]
            abs(value)>tolerance || continue
            key=(H.rowval[p],column)
            buffer=get!(acc.sum_uH,key) do
                zeros(ComplexF64,length(uv))
            end
            acc.sum_H[key]=get(acc.sum_H,key,0.0+0.0im)+value
            push!(buffers,buffer);push!(values,value)
        end
    end
    Threads.@threads for i in eachindex(buffers)
        buffer=buffers[i];value=values[i]
        @inbounds @simd for d in eachindex(uv)
            buffer[d] += uv[d]*value
        end
    end
    acc.count+=1
    push!(acc.snapshot_indices,frame);push!(acc.xdatcar_marker_labels,marker)
    return acc
end

function accumulate_snapshot!(acc::UHAccumulator,H::SparseMatrixCSC{ComplexF64,Int},u::Matrix{Float64},frame::Int,marker::Int;tolerance::Float64=0.0)
    LOCAL_ACCUMULATOR[]=acc
    MPI.bcast((action=:accumulate,H=H,u=u,frame=frame,marker=marker,tolerance=tolerance),COMM;root=0)
    local_accumulate!(acc,H,u,frame,marker,tolerance)
end

function local_restore(filename)
    # Each rank restores only its electronic columns; HDF5 reads stay outside
    # threaded regions. The on-disk format is the original combined checkpoint.
    return h5open(filename,"r") do f
        dim=Int(read(f["hamiltonian_dimension"]))
        natoms=Int(read(f["number_of_atoms"]))
        acc=UHAccumulator(dim,natoms)
        acc.count=Int(read(f["number_of_snapshots"]))
        acc.snapshot_indices=Vector{Int}(read(f["snapshot_indices"]))
        acc.xdatcar_marker_labels=Vector{Int}(read(f["xdatcar_marker_labels"]))
        rows=Vector{Int}(read(f["electronic/matrix_row_i"]))
        cols=Vector{Int}(read(f["electronic/matrix_column_j"]))
        mean_format=haskey(f,"accumulators/mean_uH_eV_angstrom")
        factor=mean_format ? acc.count : 1
        acc.sum_u .= factor .* read(f[mean_format ? "accumulators/mean_u_angstrom" : "accumulators/sum_u_angstrom"])
        hs=read(f[mean_format ? "accumulators/mean_H_eV" : "accumulators/sum_H"])
        ds=f[mean_format ? "accumulators/mean_uH_eV_angstrom" : "accumulators/sum_uH"]
        selected=findall(i->owned_column(cols[i],dim),eachindex(cols))
        for start in 1:1024:length(selected)
            indices=selected[start:min(start+1023,length(selected))]
            # Checkpoint entries are sorted by column, so rank ranges are contiguous.
            first_i,last_i=first(indices),last(indices)
            buffer=ComplexF64.(ds[:,first_i:last_i])
            for i in indices
                key=(rows[i],cols[i])
                acc.sum_H[key]=factor*hs[i]
                acc.sum_uH[key]=factor .* buffer[:,i-first_i+1]
            end
        end
        close(ds)
        LOCAL_ACCUMULATOR[]=acc
        acc
    end
end

function load_existing_accumulator(filename::AbstractString)
    MPI.bcast((action=:restore,filename=String(filename)),COMM;root=0)
    return local_restore(filename)
end

function statistics_part(acc,hs,us,mean_new,mean_old,N,M,P,ks,lo,hi)
    new2=0.0;old2=0.0;diff2=0.0;largest=0.0;components=0
    hvalues=zeros(ComplexF64,M);delta=zeros(ComplexF64,length(mean_new))
    for i in lo:hi
        key=ks[i];uh=acc.sum_uH[key]
        for t in 1:M
            h=hs[t][key[1],key[2]]
            hvalues[t]=abs(h)>H_TOLERANCE ? h : 0.0+0.0im
        end
        hnew=get(acc.sum_H,key,0.0+0.0im)/N
        hold=P>0 ? (get(acc.sum_H,key,0.0+0.0im)-sum(hvalues))/P : 0.0+0.0im
        if M>1 && P>0
            fill!(delta,0)
            for t in 1:M,d in eachindex(delta)
                delta[d]+=us[d,t]*hvalues[t]
            end
        end
        @inbounds for d in eachindex(mean_new)
            bnew=uh[d]/N-mean_new[d]*hnew
            new2+=abs2(bnew);largest=max(largest,abs(bnew))
            if P>0
                added=M==1 ? us[d,1]*hvalues[1] : delta[d]
                bold=(uh[d]-added)/P-mean_old[d]*hold
                old2+=abs2(bold);diff2+=abs2(bnew-bold)
            end
        end
        components+=length(mean_new)
    end
    return (new2,old2,diff2,largest,components)
end

function correlation_statistics(acc,monitored_H,monitored_u)
    N=acc.count;M=length(monitored_H);P=N-M
    mean_new=acc.sum_u./N
    delta_u=vec(sum(monitored_u;dims=2))
    mean_old=P>0 ? (acc.sum_u.-delta_u)./P : zeros(length(delta_u))
    ks=collect(keys(acc.sum_uH))
    nt=min(Threads.nthreads(),max(1,length(ks)))
    tasks=map(1:nt) do task
        lo=1+((task-1)*length(ks))÷nt
        hi=(task*length(ks))÷nt
        Threads.@spawn statistics_part(acc,monitored_H,monitored_u,mean_new,mean_old,N,M,P,ks,lo,hi)
    end
    partial=fetch.(tasks)
    totals=[sum(p[i] for p in partial) for i in 1:3]
    total=MPI.Allreduce(totals,+,COMM)
    largest=MPI.Allreduce(maximum(p[4] for p in partial),max,COMM)
    components=MPI.Allreduce(sum(p[5] for p in partial),+,COMM)
    return total,largest,components
end

function append_convergence_log!(acc::UHAccumulator,hs::Vector{SparseMatrixCSC{ComplexF64,Int}},us::Matrix{Float64},previous)
    MPI.bcast((action=:convergence,hs=hs,us=us),COMM;root=0)
    totals,largest,components=correlation_statistics(acc,hs,us)
    N=acc.count;M=length(hs);P=N-M
    newnorm=sqrt(totals[1]);oldnorm=sqrt(totals[2]);denom=max(newnorm,oldnorm)
    change=P>0 && denom>0 ? sqrt(totals[3])/denom : NaN
    relative=isnothing(previous) ? NaN : (max(newnorm,previous)>0 ? abs(newnorm-previous)/max(newnorm,previous) : NaN)
    rms=components>0 ? newnorm/sqrt(components) : NaN
    open(CONVERGENCE_LOG_FILE,"a") do io
        @printf(io,"%d  %d  %.12e  %.12e  %.12e  %.12e  %.12e  %d  %d\n",N,last(acc.snapshot_indices),newnorm,rms,largest,relative,change,M,P)
    end
    return newnorm
end

function local_B_norm(acc)
    localnorm=_serial_cumulative_B_norm(acc)
    return sqrt(MPI.Allreduce(isnothing(localnorm) ? 0.0 : localnorm^2,+,COMM))
end
function cumulative_B_norm(acc::UHAccumulator)
    MPI.bcast((action=:norm,),COMM;root=0)
    return local_B_norm(acc)
end

function prepare_checkpoint(acc)
    ks=_serial_sorted_electronic_keys(acc)
    CHECKPOINT_KEYS[]=ks
    parts=MPI.gather((ks,[get(acc.sum_H,k,0.0+0.0im) for k in ks]),COMM;root=0)
    if RANK==0
        CHECKPOINT_COUNTS[]=[length(p[1]) for p in parts]
        empty!(CHECKPOINT_MEAN_H)
        for (keys_rank,hs) in parts,(key,h) in zip(keys_rank,hs)
            CHECKPOINT_MEAN_H[key]=h
        end
        return reduce(vcat,[p[1] for p in parts])
    end
    return nothing
end
function sorted_electronic_keys(acc::UHAccumulator)
    MPI.bcast((action=:prepare_checkpoint,),COMM;root=0)
    return prepare_checkpoint(acc)
end

function pack_checkpoint(acc,dataset=nothing)
    ks=CHECKPOINT_KEYS[]
    width=1024;ndof=length(acc.sum_u)
    maxentries=MPI.Allreduce(length(ks),max,COMM)
    buffer=zeros(ComplexF64,ndof,width)
    receive=RANK==0 ? zeros(ComplexF64,ndof,width,NRANKS) : nothing
    offsets=RANK==0 ? cumsum(vcat(0,CHECKPOINT_COUNTS[][1:end-1])) : Int[]
    for start in 1:width:maxentries
        fill!(buffer,0)
        ncols=max(0,min(width,length(ks)-start+1))
        Threads.@threads for col in 1:ncols
            source=get(acc.sum_uH,ks[start+col-1],nothing)
            if source!==nothing
                @inbounds @simd for d in 1:ndof
                    buffer[d,col]=source[d]/acc.count
                end
            end
        end
        MPI.Gather!(buffer,receive,COMM;root=0)
        if RANK==0
            for rank in 1:NRANKS
                count=max(0,min(width,CHECKPOINT_COUNTS[][rank]-start+1))
                count==0 && continue
                firstcol=offsets[rank]+start
                dataset[:,firstcol:firstcol+count-1]=Array(@view receive[:,1:count,rank])
            end
        end
    end
end
function write_mean_uH_columns!(group,acc::UHAccumulator,keys_sorted)
    dataset=create_dataset(group,"mean_uH_eV_angstrom",ComplexF64,(length(acc.sum_u),length(keys_sorted)))
    try
        MPI.bcast((action=:write_checkpoint,),COMM;root=0)
        pack_checkpoint(acc,dataset)
    finally
        close(dataset)
    end
end

function handle_correlation_command(message)
    action=message.action
    if action==:accumulate
        if LOCAL_ACCUMULATOR[]===nothing
            LOCAL_ACCUMULATOR[]=UHAccumulator(size(message.H,1),size(message.u,2))
        end
        local_accumulate!(LOCAL_ACCUMULATOR[],message.H,message.u,message.frame,message.marker,message.tolerance)
    elseif action==:restore
        local_restore(message.filename)
    elseif action==:convergence
        correlation_statistics(LOCAL_ACCUMULATOR[],message.hs,message.us)
    elseif action==:norm
        local_B_norm(LOCAL_ACCUMULATOR[])
    elseif action==:prepare_checkpoint
        prepare_checkpoint(LOCAL_ACCUMULATOR[])
    elseif action==:write_checkpoint
        pack_checkpoint(LOCAL_ACCUMULATOR[])
    else
        error("Unknown correlation command: $action")
    end
end
