using HDF5, LinearAlgebra
include(joinpath(@__DIR__, "CalculationPaths.jl"))
using .CalculationPaths

const HBAR_J_S=1.054571817e-34
const AMU_KG=1.66053906660e-27
const ANGSTROM_M=1e-10

function options(args)
    if isempty(args) || any(in(("--help","-h")),args)
        println("""
Usage: julia compute_electron_phonon_selected_k.jl DIRECTORY [options]
  --k-reduced X Y Z    Initial k in primitive reciprocal coordinates; default Gamma (0 0 0).
  --k-index I          Instead select the 1-based index in the stored k grid.
  --output FILE        Final HDF5 filename. Default: electron_phonon_g_selected_k.h5 at Gamma;
                       electron_phonon_g_kpoint_INDEX.h5 elsewhere.
  --remove-full-g      After calculation/checking in RAM, remove full g tensors in DIRECTORY.
  --overwrite          Replace an existing selected-k result.

Reads d, phonon_modes_from_covariance.h5 and unitary_transformation_matrices.h5.
Stores all bands, phonon branches and q points for ONE initial k. No temporary files.
The requested k must belong to the existing d mesh; no interpolation is performed.
""")
        return nothing
    end
    directory=args[1]; target=[0.,0.,0.]; index=nothing; output=nothing
    explicit_target=false; remove_full=false; overwrite=false; i=2
    while i<=length(args)
        arg=args[i]
        if arg=="--k-reduced"
            i+3<=length(args) || error("--k-reduced needs three coordinates.")
            target=parse.(Float64,args[i+1:i+3]);explicit_target=true;i+=3
        elseif arg in ("--k-index","--output")
            i<length(args) || error("Missing value after $arg")
            if arg=="--k-index";index=parse(Int,args[i+1])
            else;output=abspath(expanduser(args[i+1]));end
            i+=1
        elseif arg=="--remove-full-g";remove_full=true
        elseif arg=="--overwrite";overwrite=true
        else;error("Unknown argument: $arg");end
        i+=1
    end
    explicit_target && index!==nothing && error("Use either --k-reduced or --k-index.")
    all(isfinite,target) || error("Initial k must be finite.")
    return (;data=resolve_calculation_directory(directory),target,index,output,remove_full,overwrite)
end

function main()
    settings=options(ARGS);settings===nothing && return
    data=settings.data
    dfile=joinpath(data,"electron_lattice_d_kq.h5")
    pfile=joinpath(data,"phonon_modes_from_covariance.h5")
    ufile=joinpath(data,"unitary_transformation_matrices.h5")
    for file in (dfile,pfile,ufile);isfile(file) || error("Missing input: $file");end
    d=h5open(dfile,"r") do f
        (;reps=Int.(read(f["repetitions_n1_n2_n3"])),
          norb=Int(read(f["number_of_primitive_orbitals"])),
          nalpha=Int(read(f["number_of_primitive_atoms"])),
          ki=Int.(read(f["grids/k_integer_indices"])),
          kr=Float64.(read(f["grids/k_reduced_primitive_reciprocal"])),
          kc=Float64.(read(f["grids/k_cartesian_inv_angstrom"])),
          qi=Int.(read(f["grids/q_integer_indices"])),
          qr=Float64.(read(f["grids/q_reduced_primitive_reciprocal"])),
          qc=Float64.(read(f["grids/q_cartesian_inv_angstrom"])),
          alpha=Int.(read(f["block_index/primitive_atom_alpha"])),
          component=String.(read(f["block_index/component"])),
          mu=Int.(read(f["block_index/mu"])),nu=Int.(read(f["block_index/nu"])),
          names=String.(read(f["block_index/dataset_name"])),
          snapshots=Int.(read(f["snapshot_indices"])),
          normalization=String(read(f["normalization"])),
          phase=String(read(f["phase_convention"])))
    end
    nk=size(d.ki,2);nq=size(d.qi,2)
    ik=settings.index
    if ik===nothing
        hits=[i for i in 1:nk if maximum(abs.(d.kr[:,i]-settings.target-round.(d.kr[:,i]-settings.target)))<1e-9]
        length(hits)==1 || error("Requested initial k is absent or nonunique in the stored grid.")
        ik=only(hits)
    end
    1<=ik<=nk || error("--k-index must lie in 1:$nk")
    isgamma=maximum(abs.(d.kr[:,ik]-round.(d.kr[:,ik])))<1e-9
    output=settings.output===nothing ? joinpath(data,isgamma ? "electron_phonon_g_selected_k.h5" :
        "electron_phonon_g_kpoint_"*lpad(string(ik),6,'0')*".h5") : settings.output
    output in (dfile,pfile,ufile) && error("The output cannot replace a primary input.")
    isfile(output) && !settings.overwrite && error("Output exists: $output; use --overwrite.")
    p=h5open(pfile,"r") do f
        d.reps==read(f["repetitions_n1_n2_n3"]) || error("Phonon/d repetitions differ.")
        d.qi==read(f["q_grid/integer_indices"]) || error("Phonon/d q ordering differs.")
        isapprox(d.qr,read(f["q_grid/reduced_primitive_reciprocal"]);atol=1e-12,rtol=0) || error("Phonon/d q coordinates differ.")
        (;masses=Float64.(read(f["primitive/mass_amu"])),
          freq=Float64.(read(f["phonons/frequency_THz_signed"])),
          omega2=Float64.(read(f["phonons/omega_squared_s_minus2"])),
          imaginary=read(f["phonons/is_imaginary"]).!=0,
          eigenvectors=ComplexF64.(read(f["phonons/polarization_eigenvector"])),
          temperature=Float64(read(f["temperature_K"])))
    end
    U,energies,reference=h5open(ufile,"r") do f
        d.ki==read(f["grids/k_integer_indices"]) || error("Unitary/d k ordering differs.")
        isapprox(d.kr,read(f["grids/k_reduced_primitive_reciprocal"]);atol=1e-12,rtol=0) || error("Unitary/d k coordinates differ.")
        (ComplexF64.(read(f["bands/U_mu_n_k"])),Float64.(read(f["bands/energy_eV"])),
         haskey(f,"electronic_reference_definition") ? String(read(f["electronic_reference_definition"])) : "MD-averaged Hamiltonian")
    end
    nb=size(U,2);nm=size(p.freq,1)
    size(U)==(d.norb,nb,nk) || error("Unexpected unitary dimensions.")
    size(p.eigenvectors)==(3,d.nalpha,nm,nq) || error("Unexpected polarization dimensions.")
    all(>(0),p.masses) || error("Nonpositive masses.")
    valid=(.!p.imaginary).&(p.omega2.>0).&(p.freq.>1e-6)
    lookup=Dict(Tuple(mod.(d.ki[:,i],d.reps))=>i for i in 1:nk)
    final_indices=[lookup[Tuple(mod.(d.ki[:,ik]+d.qi[:,iq],d.reps))] for iq in 1:nq]
    selected_d=zeros(ComplexF64,3*d.nalpha,d.norb,d.norb,nq)
    seen=Set{NTuple{3,Int}}()
    println("Reading d at initial k index $ik/$nk, reduced coordinates $(d.kr[:,ik]); all $nq q points.");flush(stdout)
    h5open(dfile,"r") do f
        for entry in eachindex(d.names)
            a=findfirst(==(lowercase(d.component[entry])),["x","y","z"])
            a===nothing && error("Unknown displacement component.")
            row=3*(d.alpha[entry]-1)+a;mu=d.mu[entry];nu=d.nu[entry]
            key=(row,mu,nu);key in seen && error("Duplicate d block.");push!(seen,key)
            dataset=f["d_blocks/"*d.names[entry]]
            size(dataset)==(nk,nq) || error("Unexpected d-block dimensions.")
            @views selected_d[row,mu,nu,:].=ComplexF64.(dataset[ik,:])
            close(dataset)
        end
    end
    g=fill(ComplexF64(NaN,NaN),nb,nb,nm,nq)
    Gorb=zeros(ComplexF64,d.norb,d.norb);work=zeros(ComplexF64,nb,d.norb)
    Uk=@view U[:,:,ik]
    for iq in 1:nq,mode in 1:nm
        valid[mode,iq] || continue
        fill!(Gorb,0)
        for alpha in 1:d.nalpha,a in 1:3
            factor=sqrt(HBAR_J_S/(2*p.masses[alpha]*AMU_KG*sqrt(p.omega2[mode,iq])))/ANGSTROM_M * p.eigenvectors[a,alpha,mode,iq]
            row=3*(alpha-1)+a
            for nu in 1:d.norb,mu in 1:d.norb
                Gorb[mu,nu]+=factor*selected_d[row,mu,nu,iq]
            end
        end
        mul!(work,adjoint(@view(U[:,:,final_indices[iq]])),Gorb)
        mul!(@view(g[:,:,mode,iq]),work,Uk)
    end
    # Compare one finite matrix with the previously computed orbital-g file,
    # if available, before removing any derived tensors.
    old=joinpath(data,"electron_phonon_g_kq.h5")
    if isfile(old)
        sample=findfirst(valid)
        sample===nothing && error("No valid phonon modes.")
        mode,iq=Tuple(sample)
        check=zeros(ComplexF64,d.norb,d.norb)
        h5open(old,"r") do f
            modes=Int.(read(f["block_index/mode"]));mus=Int.(read(f["block_index/mu"]))
            nus=Int.(read(f["block_index/nu"]));names=String.(read(f["block_index/dataset_name"]))
            for e in findall(==(mode),modes)
                check[mus[e],nus[e]]=f["g_blocks/"*names[e]][ik,iq]
            end
        end
        expected=U[:,:,final_indices[iq]]'*check*U[:,:,ik]
        isapprox(g[:,:,mode,iq],expected;rtol=1e-10,atol=1e-13) || error("Selected-k contraction disagrees with existing orbital-g result.")
        println("Existing-result check passed; max difference $(maximum(abs.(g[:,:,mode,iq]-expected))) eV.")
    end
    println("Calculated all $nm modes and $nb x $nb band pairs for the selected k.");flush(stdout)
    if settings.remove_full
        for name in ("electron_phonon_g_kq.h5","electron_phonon_g_kq.h5.tmp",
                     "electron_phonon_g_band_kq.h5","electron_phonon_g_band_kq.h5.tmp")
            file=joinpath(data,name)
            if isfile(file);rm(file);println("Removed derived full tensor: $file");end
        end
    end
    # Write only the final compact file directly: no .tmp, checkpoint or scratch files.
    try
        h5open(output,"w") do f
            f["format_version"]=1;f["result_kind"]="band electron-phonon couplings for one selected initial k"
            f["is_complete"]=Int8(0)
            f["number_of_k_points"]=1;f["original_number_of_k_points"]=nk
            f["selected_k_index_in_original_grid_1based"]=ik
            f["number_of_q_points"]=nq;f["number_of_modes"]=nm;f["number_of_bands"]=nb
            f["number_of_primitive_atoms"]=d.nalpha;f["number_of_primitive_orbitals"]=d.norb
            f["repetitions_n1_n2_n3"]=d.reps;f["units"]="eV";f["temperature_K"]=p.temperature
            f["number_of_snapshots"]=length(d.snapshots);f["snapshot_indices"]=d.snapshots
            f["electronic_reference_definition"]=reference
            f["formula"]="g_band(k,q) = U(k+q)' * sum_alpha,a [sqrt(hbar/(2*M_alpha*omega))*e_alpha,a*d_alpha,a(k,q)] * U(k)"
            f["g_matrix_axis_1"]="selected initial k (length 1)";f["g_matrix_axis_2"]="q index"
            f["sources/d_couplings"]=dfile;f["sources/phonons"]=pfile;f["sources/unitary_transformations"]=ufile
            f["conventions/d_normalization"]=d.normalization;f["conventions/d_phase_convention"]=d.phase
            f["conventions/polarization_conjugated"]=false
            f["grids/k_integer_indices"]=d.ki[:,ik:ik];f["grids/k_reduced_primitive_reciprocal"]=d.kr[:,ik:ik]
            f["grids/k_cartesian_inv_angstrom"]=d.kc[:,ik:ik]
            f["grids/q_integer_indices"]=d.qi;f["grids/q_reduced_primitive_reciprocal"]=d.qr
            f["grids/q_cartesian_inv_angstrom"]=d.qc
            f["grids/final_k_index_in_original_grid_1based"]=final_indices
            f["bands/initial_energy_eV"]=energies[:,ik:ik];f["bands/final_energy_eV"]=energies[:,final_indices]
            f["phonons/frequency_THz_signed"]=p.freq;f["phonons/omega_squared_s_minus2"]=p.omega2
            f["phonons/valid_for_g"]=Int8.(valid)
            modes=Int[];ms=Int[];ns=Int[];names=String[]
            blocks=create_group(f,"g_blocks")
            for mode in 1:nm,m in 1:nb,n in 1:nb
                name="block_"*lpad(string(length(names)+1),8,'0')
                blocks[name]=reshape(g[m,n,mode,:],1,nq)
                push!(modes,mode);push!(ms,m);push!(ns,n);push!(names,name)
            end
            f["block_index/mode"]=modes;f["block_index/band_m"]=ms;f["block_index/band_n"]=ns
            f["block_index/m"]=ms;f["block_index/n"]=ns;f["block_index/dataset_name"]=names
            f["block_index/number_of_blocks"]=length(names)
            write(f["is_complete"],Int8(1))
        end
    catch
        isfile(output) && rm(output)
        rethrow()
    end
    println("Saved selected-k result directly: $output ($(round(filesize(output)/1024^2;digits=2)) MiB)")
end
main()
