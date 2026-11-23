module DirectFourierKernel
using FFTW
export plus_indices, accumulate_fourier_snapshot!
cellindex(c, reps) = 1 + c[1] + reps[1]*(c[2] + reps[2]*c[3])

function plus_indices(reps)
    cells = [(r1,r2,r3) for r3 in 0:reps[3]-1 for r2 in 0:reps[2]-1 for r1 in 0:reps[1]-1]
    nc = length(cells)
    result = Matrix{Int}(undef,nc,nc)
    for q in 1:nc, k in 1:nc
        result[k,q] = cellindex(ntuple(d -> mod(cells[k][d]+cells[q][d],reps[d]),3),reps)
    end
    result
end

"""Add one snapshot's contribution to an unnormalized, one-orbital-pair d sum.
H_RR is overwritten by its Fourier transform. S has axes (k,q,ionic row).
"""
function accumulate_fourier_snapshot!(S, H_RR, Z, reps, kplusq;
                                      forward_plan=nothing, backward_plan=nothing)
    nc = prod(reps)
    size(H_RR) == (nc,nc) || error("Wrong H_RR shape.")
    size(S) == (nc,nc,size(Z,1)) || error("Wrong d block shape.")
    sixgrid = reshape(H_RR, reps..., reps...)
    if forward_plan === nothing
        fft!(sixgrid,(1,2,3))
        bfft!(sixgrid,(4,5,6))
    else
        forward_plan * sixgrid
        backward_plan * sixgrid
    end
    factor = inv(nc*sqrt(Float64(nc)))
    @inbounds for a in axes(Z,1), q in 1:nc
        weight = Z[a,q]*factor
        for k in 1:nc
            S[k,q,a] += weight*H_RR[kplusq[k,q],k]
        end
    end
    return S
end

end # module
