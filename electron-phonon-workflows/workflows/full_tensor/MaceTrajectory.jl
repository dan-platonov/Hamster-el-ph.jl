module MaceTrajectory
using LinearAlgebra, Printf
export foreach_frame, first_frame, primitive_sites, export_xdatcar
const ELEMENTS = split(get(ENV, "EPH_LAMMPS_SPECIES", "Pb,Cs,Br"), ",") # LAMMPS type 1,2,... order; set explicitly for a new model.

function read_frame(io)
    eof(io) && return nothing
    strip(readline(io)) == "ITEM: TIMESTEP" || error("Expected LAMMPS TIMESTEP header")
    step = parse(Int, readline(io))
    strip(readline(io)) == "ITEM: NUMBER OF ATOMS" || error("Missing LAMMPS atom count")
    n = parse(Int, readline(io)); n > 0 || error("Empty MD frame")
    bounds_header = split(strip(readline(io)))
    bounds_header[1:3] == ["ITEM:", "BOX", "BOUNDS"] || error("Missing BOX BOUNDS")
    bounds_header[4:end] == ["pp", "pp", "pp"] || error("This reader requires a periodic orthorhombic box")
    bounds = [parse.(Float64, split(readline(io))) for _ in 1:3]
    all(length(b) == 2 for b in bounds) || error("Unsupported tilted box")
    origin = [b[1] for b in bounds]
    lengths = [b[2]-b[1] for b in bounds]
    all(>(0), lengths) || error("Nonpositive box dimension")
    lattice = Matrix(Diagonal(lengths))
    fields = split(strip(readline(io)))
    fields[1:2] == ["ITEM:", "ATOMS"] || error("Missing ATOMS header")
    columns = String.(fields[3:end])
    idcol = findfirst(==("id"), columns); typecol = findfirst(==("type"), columns)
    idcol !== nothing && typecol !== nothing || error("Atom id/type columns are required")
    xyz = findfirst.([==("x"), ==("y"), ==("z")], Ref(columns))
    scaled = false
    if any(isnothing, xyz)
        xyz = findfirst.([==("xs"), ==("ys"), ==("zs")], Ref(columns)); scaled = true
    end
    any(isnothing, xyz) && error("Expected Cartesian x y z or scaled xs ys zs")
    ids = zeros(Int,n); types = zeros(Int,n); frac = zeros(3,n)
    for atom in 1:n
        eof(io) && error("Truncated MD frame at step $step")
        values = split(readline(io))
        length(values) == length(columns) || error("Invalid atom row at step $step")
        ids[atom] = parse(Int,values[idcol]); types[atom] = parse(Int,values[typecol])
        1 <= types[atom] <= length(ELEMENTS) || error("Unknown atom type")
        for a in 1:3
            x = parse(Float64,values[xyz[a]])
            frac[a,atom] = scaled ? x : (x-origin[a])/lengths[a]
        end
    end
    length(unique(ids)) == n || error("Duplicate atom ids")
    all(isfinite, frac) || error("Nonfinite MD coordinates")
    # Species grouping is required by POSCAR; atom identity stays fixed by id.
    order = sortperm(1:n; by=i -> (types[i], ids[i]))
    ids=ids[order]; types=types[order]; frac=mod.(frac[:,order],1.0)
    names = [ELEMENTS[t] for t in sort(unique(types))]
    counts = [count(==(name),ELEMENTS[types]) for name in names]
    return (;step, ids, types, fractional=frac, lattice, origin,
            species=names, counts, atom_species=ELEMENTS[types])
end

first_frame(filename) = open(read_frame, filename, "r")

function foreach_frame(callback, filename, start=1, final=nothing)
    selected=0; ordinal=0; reference=nothing; previous_step=nothing
    open(filename,"r") do io
        while !eof(io) && (final === nothing || ordinal < final)
            frame=read_frame(io); frame === nothing && break; ordinal+=1
            if reference === nothing
                reference=frame
            else
                frame.ids == reference.ids && frame.types == reference.types || error("Atom identity/type changed")
                isapprox(frame.lattice,reference.lattice;atol=1e-10,rtol=1e-10) || error("Variable-cell MD is unsupported")
                isapprox(frame.origin,reference.origin;atol=1e-10,rtol=0) || error("Box origin changed")
                frame.step > previous_step || error("Non-increasing LAMMPS timesteps")
            end
            previous_step=frame.step
            if ordinal >= start
                callback(ordinal,frame); selected+=1
            end
        end
    end
    return selected,ordinal
end

function primitive_sites(frame, reps; tolerance_angstrom=1e-3)
    nc=prod(reps); length(frame.ids)%nc == 0 || error("Atom count incompatible with repetitions")
    lattice=frame.lattice*Diagonal(1.0 ./ collect(reps))
    sites=Vector{Vector{Float64}}(); labels=String[]; multiplicity=Int[]
    for atom in eachindex(frame.ids)
        folded=mod.(frame.fractional[:,atom].*collect(reps),1.0)
        species=frame.atom_species[atom]
        hit=findfirst(eachindex(sites)) do j
            delta=folded-sites[j]; delta.-=round.(delta)
            labels[j] == species && norm(lattice*delta) < tolerance_angstrom
        end
        if hit === nothing
            push!(sites,folded); push!(labels,species); push!(multiplicity,1)
        else
            delta=folded-sites[hit]; delta.-=round.(delta)
            multiplicity[hit]+=1
            sites[hit].=mod.(sites[hit].+delta./multiplicity[hit],1.0)
        end
    end
    all(==(nc),multiplicity) || error("First frame is not primitive-periodic; supply an explicit --primitive POSCAR")
    length(sites)==div(length(frame.ids),nc) || error("Wrong primitive site count")
    return (;lattice, fractional=hcat(sites...), species=frame.species,
            counts=[count(==(s),labels) for s in frame.species], atom_species=labels)
end

function print_header(io, frame)
    println(io,"CsPbBr3 MACE trajectory; species/type order Pb Cs Br")
    println(io,"1.0")
    for a in 1:3
        @printf(io," %.14f %.14f %.14f\n",frame.lattice[1,a],frame.lattice[2,a],frame.lattice[3,a])
    end
    println(io,join(frame.species," ")); println(io,join(frame.counts," "))
end

function export_xdatcar(source,destination; overwrite=true)
    !overwrite && ispath(destination) && error("Output exists: $destination")
    temp=destination*".tmp"
    try
        open(temp,"w") do io
            foreach_frame(source) do ordinal,frame
                ordinal==1 && print_header(io,frame)
                # Configuration labels match one-based frame ordinals used by HAMSTER.
                println(io,"Direct configuration= ",ordinal)
                for atom in eachindex(frame.ids)
                    @printf(io," %.14f %.14f %.14f\n",frame.fractional[1,atom],frame.fractional[2,atom],frame.fractional[3,atom])
                end
            end
        end
        mv(temp,destination;force=overwrite)
    catch
        isfile(temp) && rm(temp)
        rethrow()
    end
end
end
