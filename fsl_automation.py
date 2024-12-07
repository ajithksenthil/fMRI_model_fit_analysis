import os
import glob
import subprocess
import shutil

# ---------------------- USER-DEFINED SETTINGS ---------------------- #
# Adjust this base directory to your main "analysis" folder
BASE_DIR = "/Users/ajithsenthil/Desktop/Psych_555/analysis"
BIDS_DIR = os.path.join(BASE_DIR, "bids_dataset")

# Directory holding timing files
TIMING_DIR = os.path.join(BASE_DIR, "timing_files")

# Directory to store the .feat outputs and fsf files
MODEL_COMPARISON_DIR = os.path.join(BASE_DIR, "model_comparison")
os.makedirs(MODEL_COMPARISON_DIR, exist_ok=True)

# Standard space template
STANDARD_IMAGE = "/Users/ajithsenthil/fsl/data/standard/MNI152_T1_2mm_brain"

# Scanner/sequence parameters as given
TR = 1.0
N_VOLS = 360
EPI_DWELL_TIME = 0.000579999
EPI_TE = 35
UNWARP_DIR = "y-"
SMOOTH_FWHM = 5
HP_CUTOFF = 100
Z_THRESH = 2.3
CLUSTER_P = 0.05
DELTA_TE = 2.46  # fieldmap delta TE for fsl_prepare_fieldmap

# EV names as per instructions
EV_NAMES = ["noise0", "noise25", "noise50", "noise75", "noise100"]

# Contrast arrays as per instructions
linear_clarity = [1.00, 0.75, 0.50, 0.25, 0.00]
quadratic_clarity = [1.000, 0.563, 0.250, 0.063, 0.000]
cubic_clarity = [1.000, 0.422, 0.125, 0.016, 0.000]
exponential_clarity = [1.000, 0.865, 0.632, 0.393, 0.000]
control_clarity = [1.0, 1.0, 1.0, 1.0, 1.0]

CONTRAST_NAMES = ["linear", "quadratic", "cubic", "exponential", "control"]
CONTRASTS = [linear_clarity, quadratic_clarity, cubic_clarity, exponential_clarity, control_clarity]

# Registration parameters
REGISTRATION_SEARCH = 90
REGISTRATION_DOF = "BBR"  # Brain Boundary Registration

# ---------------------- FUNCTIONS ---------------------- #

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def find_subjects(bids_dir):
    # Identify directories starting with sub-
    subjects = glob.glob(os.path.join(bids_dir, "sub-*"))
    return [os.path.basename(s) for s in subjects]

def find_functional_runs(sub_dir):
    # Find all BOLD runs
    func_dir = os.path.join(sub_dir, "ses-Psych555GroupProjectsF24", "func")
    runs = sorted(glob.glob(os.path.join(func_dir, "*_bold.nii.gz")))
    return runs

def prepare_fieldmap(sub_dir):
    # Locate fieldmap images
    fmap_dir = os.path.join(sub_dir, "ses-Psych555GroupProjectsF24", "fmap")
    phasediff = glob.glob(os.path.join(fmap_dir, "*run-01*_phasediff.nii.gz"))[0]
    magnitude1 = glob.glob(os.path.join(fmap_dir, "*run-01*_magnitude1.nii.gz"))[0]

    # Brain extract the magnitude image
    mag_brain = magnitude1.replace(".nii.gz", "_brain.nii.gz")
    if not os.path.exists(mag_brain):
        run_command(f"bet {magnitude1} {mag_brain} -B -f 0.7")

    # Prepare fieldmap in radians
    phase_rads = phasediff.replace("_phasediff", "_phase_rads")
    if not os.path.exists(phase_rads + ".nii.gz"):
        run_command(f"fsl_prepare_fieldmap SIEMENS {phasediff} {mag_brain} {phase_rads} {DELTA_TE}")

    return mag_brain, phase_rads  # Remove the + ".nii.gz" here since fsl_prepare_fieldmap already adds it

def brain_extract_anat(sub_dir):
    # Brain extraction for T1 image
    anat_dir = os.path.join(sub_dir, "ses-Psych555GroupProjectsF24", "anat", "derivatives")
    t1 = glob.glob(os.path.join(anat_dir, "*_t1w.nii.gz"))[0]
    t1_brain = t1.replace(".nii.gz", "_brain.nii.gz")
    if not os.path.exists(t1_brain):
        run_command(f"bet {t1} {t1_brain} -f 0.3 -g 0")
    return t1_brain

def create_fsf_file(output_dir, func_file, mag_brain, phase_rads, t1_brain, timing_files):
    fsf_content = []
    
    # Basic FEAT setup with exact parameters matching the working example
    fsf_content.extend([
        "# FEAT version number",
        "set fmri(version) 6.00",
        "set fmri(inmelodic) 0",
        "set fmri(level) 1",
        "set fmri(analysis) 7",
        "set fmri(relative_yn) 0",
        "set fmri(help_yn) 1",
        "set fmri(featwatcher_yn) 1",
        "set fmri(sscleanup_yn) 0",
    ])

    # Critical parameters exactly as in working example
    fsf_content.extend([
        f"set fmri(outputdir) \"{output_dir}\"",
        f"set fmri(tr) {TR}",
        f"set fmri(npts) {N_VOLS}",
        "set fmri(ndelete) 0",
        "set fmri(tagfirst) 1",
        "set fmri(multiple) 1",
        "set fmri(inputtype) 2",
    ])

    # Pre-processing parameters
    fsf_content.extend([
        "set fmri(filtering_yn) 1",
        "set fmri(brain_thresh) 10",
        "set fmri(critical_z) 5.3",
        "set fmri(noise) 0.66",
        "set fmri(noisear) 0.34",
        "set fmri(mc) 1",
        "set fmri(sh_yn) 0",
        "set fmri(regunwarp_yn) 1",
        f"set fmri(dwell) {EPI_DWELL_TIME}",
        f"set fmri(te) {EPI_TE}",
        "set fmri(signallossthresh) 10",
        f"set fmri(unwarp_dir) {UNWARP_DIR}",
        # Add the missing high-pass filter parameter
        f"set fmri(paradigm_hp) {HP_CUTOFF}",
    ])

    # Add registration parameters
    fsf_content.extend([
        # Registration parameters
        "set fmri(reginitial_highres_yn) 0",
        f"set fmri(reginitial_highres_search) {REGISTRATION_SEARCH}",
        "set fmri(reginitial_highres_dof) 3",
        
        "set fmri(reghighres_yn) 1",
        f"set fmri(reghighres_search) {REGISTRATION_SEARCH}",
        f"set fmri(reghighres_dof) {REGISTRATION_DOF}",
        
        "set fmri(regstandard_yn) 1",
        "set fmri(alternateReference_yn) 0",
        f"set fmri(regstandard) \"{STANDARD_IMAGE}\"",
        f"set fmri(regstandard_search) {REGISTRATION_SEARCH}",
        "set fmri(regstandard_dof) 12",
        "set fmri(regstandard_nonlinear_yn) 0",
        "set fmri(regstandard_nonlinear_warpres) 10",
        
        # Critical parameters
        "set fmri(totalVoxels) 79902720",
        "set fmri(fnirt_config) \"T1_2_MNI152_2mm\"",
    ])

    # Key model parameters
    fsf_content.extend([
        "set fmri(st) 0",
        "set fmri(bet_yn) 1",
        f"set fmri(smooth) {SMOOTH_FWHM}",
        "set fmri(norm_yn) 0",
        "set fmri(temphp_yn) 1",
        "set fmri(templp_yn) 0",
        "set fmri(melodic_yn) 0",
        "set fmri(stats_yn) 1",
        "set fmri(prewhiten_yn) 1",
        "set fmri(motionevs) 0",
        "set fmri(robust_yn) 0",
        "set fmri(mixed_yn) 2",
        "set fmri(evs_orig) 5",
        "set fmri(evs_real) 10",
        "set fmri(evs_vox) 0",
        "set fmri(ncon_orig) 5",
        "set fmri(ncon_real) 5",
        "set fmri(nftests_orig) 0",
        "set fmri(nftests_real) 0",
    ])

    # Files setup
    fsf_content.extend([
        f"set feat_files(1) \"{func_file}\"",
        f"set unwarp_files(1) \"{mag_brain}\"",
        f"set unwarp_files_mag(1) \"{phase_rads}\"",
        f"set highres_files(1) \"{t1_brain}\"",
    ])

    # EVs setup - critically different from our previous implementation
    for i, (ev_name, timing_file) in enumerate(zip(EV_NAMES, timing_files), start=1):
        fsf_content.extend([
            f"set fmri(evtitle{i}) \"{ev_name}\"",
            f"set fmri(shape{i}) 3",
            f"set fmri(convolve{i}) 3",  # Double-Gamma HRF
            f"set fmri(convolve_phase{i}) 0",
            f"set fmri(tempfilt_yn{i}) 1",
            f"set fmri(deriv_yn{i}) 1",
            f"set fmri(custom{i}) \"{timing_file}\"",
        ])
        
        # Orthogonalization - crucial for model stability
        for j in range(6):  # Including 0 for reference
            fsf_content.append(f"set fmri(ortho{i}.{j}) 0")

    # Contrast setup - this was a key missing piece
    fsf_content.extend([
        "set fmri(con_mode_old) orig",
        "set fmri(con_mode) orig",
    ])

    # Set up both real and orig contrasts as in the example
    for ci, (cname, contrast) in enumerate(zip(CONTRAST_NAMES, CONTRASTS), start=1):
        # Real contrasts
        fsf_content.extend([
            f"set fmri(conpic_real.{ci}) 1",
            f"set fmri(conname_real.{ci}) \"{cname}\"",
        ])
        
        # Real contrasts for both main EVs and temporal derivatives
        for evi, val in enumerate(contrast, start=1):
            fsf_content.append(f"set fmri(con_real{ci}.{evi}) {val}")
            fsf_content.append(f"set fmri(con_real{ci}.{evi+5}) 0")  # Temporal derivatives
            
        # Original contrasts
        fsf_content.extend([
            f"set fmri(conpic_orig.{ci}) 1",
            f"set fmri(conname_orig.{ci}) \"{cname}\"",
        ])
        
        for evi, val in enumerate(contrast, start=1):
            fsf_content.append(f"set fmri(con_orig{ci}.{evi}) {val}")

    # Contrast masking setup
    fsf_content.append("set fmri(conmask_zerothresh_yn) 0")
    for i in range(1, 6):
        for j in range(1, 6):
            fsf_content.append(f"set fmri(conmask{i}_{j}) 0")

    # Final parameters
    fsf_content.extend([
        "set fmri(alternative_mask) \"\"",
        "set fmri(init_initial_highres) \"\"",
        "set fmri(init_highres) \"\"",
        "set fmri(init_standard) \"\"",
        "set fmri(overwrite_yn) 0",
    ])

    return "\n".join(fsf_content)


def main():
    subjects = find_subjects(BIDS_DIR)

    for sub in subjects:
        sub_dir = os.path.join(BIDS_DIR, sub)
        # Prepare fieldmap
        mag_brain, phase_rads = prepare_fieldmap(sub_dir)
        # Prepare anatomical
        t1_brain = brain_extract_anat(sub_dir)
        # Find functional runs
        runs = find_functional_runs(sub_dir)

        # For each run, create an fsf and run feat
        for run_file in runs:
            run_base = os.path.basename(run_file).replace("_bold.nii.gz", "")
            # Example run_base: sub-GAS1_ses-Psych555GroupProjectsF24_task-task1_run-01

            timing_files = []
            for ev_name in EV_NAMES:
                level_str = ev_name.replace("noise", "noiselevel")
                # Extract subject ID and run number for timing file path
                subject_id = run_base.split('_')[0].replace('sub-', '')
                run_num = run_base.split('run-')[1][:2].zfill(3)
                timing_file = os.path.join(TIMING_DIR, f"{subject_id}_Stimuli_{run_num}_{level_str}_timingfile.txt")
                timing_files.append(timing_file)

            output_dir = os.path.join(MODEL_COMPARISON_DIR, f"{run_base}.feat")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            fsf_text = create_fsf_file(output_dir, run_file, mag_brain, phase_rads, t1_brain, timing_files)
            fsf_file = os.path.join(MODEL_COMPARISON_DIR, f"{run_base}.fsf")
            with open(fsf_file, 'w') as f:
                f.write(fsf_text)

            # Run FEAT
            run_command(f"feat {fsf_file}")


if __name__ == "__main__":
    main()
