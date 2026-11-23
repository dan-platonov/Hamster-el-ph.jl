#!/usr/bin/env python3
"""Export physical d=sum_d/N from completed direct-Fourier pair files.

Usage: /usr/bin/python3 export_direct_pair_d.py DATA_DIRECTORY
Requires phonon_modes_from_covariance.h5 for the established commensurate grid.
Reads the pair files only; does not run HAMSTER or modify any checkpoint.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = args.directory.expanduser().resolve()
    work = data / ".direct_eph_work"
    output = data / "electron_lattice_d_kq.h5"
    temporary = Path(str(output) + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Existing output or partial output: {output}")
    with h5py.File(work/"pair_0001_0001.h5", "r") as f:
        reps = f["mapping/reps"][()].astype(int)
        norb = int(f["mapping/norb"][()])
        nalpha = int(f["mapping/nalpha"][()])
        count = int(f["number_of_snapshots"][()])
        signature = f["signature"][()]
        window = int(f["completed_window"][()])
        last_frame = int(f["last_frame"][()])
        ionic_dof = f["mapping/ionic_dof"][()].astype(int)-1
    nc = int(np.prod(reps))
    cells = np.array([(i,j,k) for k in range(reps[2])
                      for j in range(reps[1]) for i in range(reps[0])])
    with h5py.File(data/"displacement_covariance.h5", "r") as c:
        snapshots = c["snapshot_indices"][()]
        if count != len(snapshots) or count <= 0 or last_frame != snapshots[-1]:
            raise ValueError("Pair files do not cover the complete covariance snapshot set.")
        if not np.array_equal(reps,c["mapping/repetitions_n1_n2_n3"][()]):
            raise ValueError("Pair/covariance supercell mismatch.")
        if ionic_dof.shape != (nc,3*nalpha):
            raise ValueError("Unexpected ionic mapping shape.")
        alpha = c["dof/primitive_atom_alpha"][()]
        labels = c["dof/cell_L"][()]
        components = c["dof/component"][()]
        for a in range(3*nalpha):
            selected = ionic_dof[:,a]
            if not (np.all(alpha[selected]==a//3+1) and
                    np.array_equal(labels[selected],cells) and
                    all((x.decode() if isinstance(x,bytes) else str(x))=="xyz"[a%3]
                        for x in components[selected])):
                raise ValueError("Pair/covariance ionic labels differ.")
    for mu in range(1,norb+1):
        for nu in range(1,norb+1):
            path=work/f"pair_{mu:04d}_{nu:04d}.h5"
            with h5py.File(path,"r") as f:
                expected={"number_of_snapshots":count,"completed_window":window,
                          "last_frame":last_frame,"signature":signature,
                          "primitive_orbital_mu":mu,"primitive_orbital_nu":nu}
                if any(f[key][()]!=value for key,value in expected.items()):
                    raise ValueError(f"Inconsistent checkpoint: {path}")
                if f["sum_d"].shape!=(3*nalpha,nc,nc):
                    raise ValueError(f"Unexpected tensor dimensions: {path}")
    print(f"Validated {norb*norb} pairs; {count} snapshots; repetitions {reps.tolist()}",flush=True)
    text=h5py.string_dtype("utf-8")
    with h5py.File(data/"phonon_modes_from_covariance.h5","r") as p,h5py.File(temporary,"w") as out:
        if not np.array_equal(p["q_grid/integer_indices"][()],cells):
            raise ValueError("Phonon grid ordering differs from the pair-file convention.")
        out["format_version"]=2
        out["number_of_snapshots"]=count
        out["snapshot_indices"]=snapshots
        out["number_of_primitive_cells"]=nc
        out["number_of_primitive_atoms"]=nalpha
        out["number_of_primitive_orbitals"]=norb
        out["repetitions_n1_n2_n3"]=reps
        out["normalization"]="1/sqrt(Nc)"
        out["phase_convention"]="exp[-2pi*i*(k+q).R] exp[+2pi*i*k.Rprime]"
        out["sources/direct_pair_directory"]=str(work)
        out["sources/displacement_covariance"]=str(data/"displacement_covariance.h5")
        out["sources/export_definition"]="d = sum_d / number_of_snapshots; no additional cell factor"
        out["sources/checkpoint_signature"]=signature
        for axis in ("k","q"):
            for field in ("integer_indices","reduced_primitive_reciprocal","cartesian_inv_angstrom"):
                out[f"grids/{axis}_{field}"]=p[f"q_grid/{field}"][()]
        out["grids/d_matrix_axis_1"]="k index"
        out["grids/d_matrix_axis_2"]="q index"
        entries=[(a,mu,nu) for a in range(3*nalpha)
                 for mu in range(1,norb+1) for nu in range(1,norb+1)]
        names=[f"block_{i:08d}" for i in range(1,len(entries)+1)]
        out["block_index/primitive_atom_alpha"]=[a//3+1 for a,mu,nu in entries]
        out.create_dataset("block_index/component",data=["xyz"[a%3] for a,mu,nu in entries],dtype=text)
        out["block_index/mu"]=[mu for a,mu,nu in entries]
        out["block_index/nu"]=[nu for a,mu,nu in entries]
        out.create_dataset("block_index/dataset_name",data=names,dtype=text)
        out["block_index/number_of_blocks"]=len(entries)
        blocks=out.create_group("d_blocks")
        for mu in range(1,norb+1):
            for nu in range(1,norb+1):
                with h5py.File(work/f"pair_{mu:04d}_{nu:04d}.h5","r") as f:
                    for a in range(3*nalpha):
                        index=a*norb*norb+(mu-1)*norb+nu-1
                        values=f["sum_d"][a,:,:]/count
                        blocks[names[index]]=values
                print(f"Exported pair ({mu},{nu})/{norb}x{norb}",flush=True)
        out["is_complete"]=True
    temporary.replace(output)
    print(f"Saved physical d tensor: {output}",flush=True)


if __name__=="__main__":
    main()
