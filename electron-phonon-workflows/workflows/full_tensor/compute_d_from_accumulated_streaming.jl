const NOBACKUP_DIR = ENV["EP_DATA_DIR"]
mkpath(NOBACKUP_DIR)

include(joinpath(@__DIR__, "ElectronPhononCouplingTools.jl"))
using .ElectronPhononCouplingTools
# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# Heavy input and output files
const ACCUMULATED_HAMILTONIAN_FILE =
    joinpath(NOBACKUP_DIR, "sum_uH_primitive_labels.h5")

const COVARIANCE_FILE =
    joinpath(NOBACKUP_DIR, "displacement_covariance.h5")

const INTERMEDIATE_D_FILE =
    joinpath(NOBACKUP_DIR, "electron_lattice_D.h5")

const OUTPUT_d_FILE =
    joinpath(NOBACKUP_DIR, "electron_lattice_d_kq.h5")

const PSEUDOINVERSE_RTOL = 1.0e-10
const PSEUDOINVERSE_ATOL_ANGSTROM2 = 0.0

# Keep false for the memory-efficient calculation.  The translation-averaged D
# is still fully saved; only the enormous unaveraged D_(alpha,L) matrix is not.
const SAVE_RAW_D_ALPHA_L = false

# Number of electronic Hamiltonian entries processed at once during C^+ B.
# Reduce this (e.g. 256 or 128) if RAM is very limited; increase it if desired.
const D_ELECTRONIC_BLOCK_SIZE = 512

const SHOW_D_PROGRESS = true
const FOURIER_ZERO_TOLERANCE = 0.0


function main()
    D_result = compute_translation_averaged_D(
        ACCUMULATED_HAMILTONIAN_FILE,
        COVARIANCE_FILE;
        pseudoinverse_relative_cutoff=PSEUDOINVERSE_RTOL,
        pseudoinverse_absolute_cutoff=PSEUDOINVERSE_ATOL_ANGSTROM2,
        electronic_block_size=D_ELECTRONIC_BLOCK_SIZE,
        materialize_raw_D=SAVE_RAW_D_ALPHA_L,
        show_progress=SHOW_D_PROGRESS,
    )

    save_translation_averaged_D(
        INTERMEDIATE_D_FILE,
        D_result;
        save_raw_D=SAVE_RAW_D_ALPHA_L,
    )
    println("Saved intermediate real-space D: $INTERMEDIATE_D_FILE")
    println("Covariance rank: $(D_result.covariance_rank)")
    println("D array size: $(size(D_result.D_values))")

    stored_D = load_translation_averaged_D(INTERMEDIATE_D_FILE)
    save_fourier_couplings(
        OUTPUT_d_FILE,
        stored_D;
        zero_tolerance=FOURIER_ZERO_TOLERANCE,
    )

    println("Saved momentum-resolved d(k+q,k): $OUTPUT_d_FILE")
    return nothing
end

main()
