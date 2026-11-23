using HDF5, LinearAlgebra, SparseArrays, Dates
include(joinpath(@__DIR__, "CalculationPaths.jl"))
using .CalculationPaths
include(joinpath(@__DIR__, "HamsterHrLoader.jl"))
include(joinpath(@__DIR__, "PrimitiveBasisRelabel.jl"))
const HR=HamsterHrLoader
const PB=PrimitiveBasisRelabel

function write_static_config(template, output, structure, hamfile, source)
    replacements=Dict("poscar"=>structure,"ham_file"=>hamfile,
        "write_hr"=>"true","write_hk"=>"false","skip_diag"=>"true",
        "rllm_file"=>joinpath(source,"rllm.dat"),
        "init_params"=>joinpath(source,"params.dat"),
        "kpoints"=>joinpath(source,"EIGENVAL"))
    block=""
    seen=Set{String}()
    open(output,"w") do io
        for line in readlines(template)
            stripped=strip(line)
            matched=match(r"^begin\s+([A-Za-z_][A-Za-z0-9_]*)"i,stripped)
            if matched!==nothing
                block=lowercase(matched.captures[1])
                block=="supercell" && continue
            end
            if lowercase(stripped)=="end"
                if block=="supercell"
                    block=""
                    continue
                elseif block=="options"
                    for (key,value) in replacements
                        key in seen || println(io,"    $key = $value")
                    end
                end
                block=""
            end
            block=="supercell" && continue
            tag=match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=",line)
            if tag!==nothing && block!="options" && lowercase(tag.captures[1]) in ("init_params","rllm_file")
                key=lowercase(tag.captures[1])
                value=strip(first(split(split(line,'=';limit=2)[2],'#';limit=2)))
                resolved=isabspath(value) ? value : joinpath(source,value)
                println(io,"    $key = $resolved")
                continue
            end
            if block=="options" && tag!==nothing && haskey(replacements,lowercase(tag.captures[1]))
                key=lowercase(tag.captures[1])
                println(io,"    $key = $(replacements[key])")
                push!(seen,key)
            else
                println(io,line)
            end
        end
    end
end

function main()
    if length(ARGS)!=1 || ARGS[1] in ("--help","-h")
        println("Usage: julia compute_hamiltonian_at_average_positions.jl SOURCE_DIRECTORY")
        println("Compute and save ONE static Hamiltonian at POSCAR_equlibr. Run band conversion and plotting separately.")
        return
    end
    source=realpath(expanduser(ARGS[1]))
    data=resolve_calculation_directory(source)
    average=joinpath(source,"POSCAR_equlibr")
    output=joinpath(data,"hamiltonian_at_average_positions.h5")
    isfile(output) && error("Hamiltonian reference already exists: $output")
    work=joinpath(data,".average_positions_hamiltonian")
    mkpath(work)
    config=joinpath(work,"hconf")
    hamfile=joinpath(data,"hamiltonian_at_average_positions_hr.h5")
    if !isfile(hamfile)
        write_static_config(joinpath(source,"hconf"),config,average,hamfile,source)
        println("[$(Dates.now())] Computing ONE static Hamiltonian at $average")
        flush(stdout)
        command=Cmd(Cmd([something(Sys.which("hamster"),"hamster"),"--conf",config]);dir=work)
        run(addenv(command,"JULIA_NUM_THREADS"=>"1","OMP_NUM_THREADS"=>"1","OPENBLAS_NUM_THREADS"=>"1"))
    else
        println("Reusing the existing static HAMSTER output: $hamfile")
    end
    groups=HR.available_hamster_snapshots(hamfile)
    length(groups)==1 || error("Expected exactly one static Hamiltonian group.")
    snapshot=HR.load_hamster_snapshot(hamfile,only(groups).snapshot_index)
    anchor=joinpath(data,".direct_eph_work","pair_0001_0001.h5")
    reps,mu,R,basis,norb=h5open(anchor,"r") do f
        (Tuple(Int.(read(f["mapping/reps"]))),Int.(read(f["mapping/mu"])),
         Int.(read(f["mapping/R"])),read(f["mapping/basis"]),Int(read(f["mapping/norb"])))
    end
    snapshot.dim==length(basis) && isequal(snapshot.basis,basis) || error("Static Hamiltonian and d use different bases.")
    cells=hcat([Int[i,j,k] for k in 0:reps[3]-1 for j in 0:reps[2]-1 for i in 0:reps[1]-1]...)
    nc=prod(reps)
    h=zeros(ComplexF64,norb,norb,nc)
    H=snapshot.H_supercell_pbc
    for j in 1:size(H,2), pointer in nzrange(H,j)
        value=H.nzval[pointer]
        abs(value)>1e-14 || continue
        i=H.rowval[pointer]
        delta=ntuple(a->mod(cells[a,R[j]]-cells[a,R[i]],reps[a]),3)
        index=1+delta[1]+reps[1]*(delta[2]+reps[2]*delta[3])
        h[mu[i],mu[j],index]+=value
    end
    h./=nc
    mapping=PB.build_primitive_basis_map(joinpath(source,"POSCAR_equil_primitive"),
        average,joinpath(source,"hconf"),basis,length(basis);
        lattice_tolerance=1e-7,position_tolerance_angstrom=1e-3)
    table=PB.primitive_basis_mapping_table(mapping)
    mapping.repetitions==reps && table.primitive_orbital==mu || error("Static primitive orbital mapping differs.")
    all(table.cell[:,i]==cells[:,R[i]] for i in eachindex(R)) || error("Static cell labels differ from d.")
    supercell_lattice=PB.read_simple_poscar(average).lattice
    geometry_ids=h5open(joinpath(data,"electron_lattice_d_kq.h5"),"r") do f
        read(f["snapshot_indices"])
    end
    temporary=output*".tmp"
    h5open(temporary,"w") do f
        f["format_version"]=1
        f["number_of_snapshots"]=1
        f["snapshot_indices"]=[snapshot.snapshot_index]
        f["electronic_reference_definition"]="Single Hamiltonian at primitive-periodic MD-averaged atomic positions"
        f["number_of_primitive_cells"]=nc
        f["number_of_primitive_orbitals"]=norb
        f["repetitions_n1_n2_n3"]=collect(reps)
        f["geometry/lattice_angstrom"]=supercell_lattice
        f["translation_averaged_H/h_mu_nu_delta_eV"]=h
        f["translation_averaged_H/delta_cell_integer"]=cells
        f["translation_averaged_H/definition"]="h(mu,nu,Delta)=sum_R H_at_average_positions(mu,R;nu,R+Delta)/Nc"
        g=create_group(f,"primitive_basis_mapping")
        for (name,value) in pairs(table)
            key=name==:primitive_orbital ? "primitive_orbital_mu" : name==:cell ? "cell_R" : string(name)
            g[key]=value
        end
        f["sources/average_positions_poscar"]=average
        f["sources/geometry_average_snapshot_indices"]=geometry_ids
        f["sources/static_hamster_hr"]=hamfile
        f["sources/static_hconf"]=config
        f["sources/direct_pair_mapping"]=anchor
        f["hamiltonian_entry_tolerance_eV"]=1e-14
    end
    mv(temporary,output;force=false)
    println("[$(Dates.now())] Saved Hamiltonian at averaged positions: $output")
    println("Reference saved. Next run unitary_transformation_matrix_construction.jl --directory DATA, then compute_electron_phonon_selected_k.jl DATA.")
end
main()
