#!/usr/bin/env python3
"""
run_dynagan_4dct.py

1) Finds a DICOM CT series in <input_dir>, converts it to a NIfTI.
2) Resamples to 128³ (as expected by Dynagan).
3) Runs dynagan/test_3D.py to generate a 4D-CT over alpha∈[0,1] step=0.1.
4) Extracts each phase from the 4D-NIfTI, and writes them back out
   as DICOM series into <output_dir>/phase_*/ with new UIDs etc.
5) Cleans up all temp files.
"""
from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, module="torch")
import os
import sys
import copy
import torch
import argparse
import tempfile
import subprocess
import shutil
import glob
from collections import defaultdict

import dicom2nifti
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pydicom
from util.spatialTransform import SpatialTransformer
from scipy import ndimage

def find_dicom_series(input_dir):
    """
    1) Find all CT DICOM files under input_dir
    2) Group by SeriesInstanceUID
    3) Pick the series with the most slices (and ≥3)
    4) Copy that series into a temp folder
    5) Convert to NIfTI
    6) Return the path to the best NIfTI + the temp dirs
    """
    # 1–2) collect CT files by series
    series = defaultdict(list)
    for fn in glob.glob(os.path.join(input_dir, "**", "*.dcm"),
                        recursive=True):
        try:
            ds = pydicom.dcmread(fn, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") == "CT":
            series[ds.SeriesInstanceUID].append(fn)

    if not series:
        raise RuntimeError(f"No CT series found in {input_dir!r}")

    # 3) pick the largest series
    best_uid = max(series, key=lambda u: len(series[u]))
    files = series[best_uid]
    if len(files) < 3:
        raise RuntimeError(f"Selected CT series has only {len(files)} slices")

    # 4) copy into a temp DICOM folder
    temp_dcm = tempfile.mkdtemp(prefix="dynagan_ct_")
    for i, src in enumerate(sorted(files)):
        dst = os.path.join(temp_dcm, f"{i:04d}.dcm")
        shutil.copy2(src, dst)

    # 5) convert that folder to NIfTI
    temp_nifti = tempfile.mkdtemp(prefix="dynagan_nifti_")
    dicom2nifti.convert_directory(
        temp_dcm,
        temp_nifti,
        compression=True,
        reorient=True     # for v2.6.0
    )

    # 6) pick the NIfTI with ≥3 slices (just in case)
    candidates = []
    for f in glob.glob(os.path.join(temp_nifti, "*.nii*")):
        img = nib.load(f)
        if img.header.get_data_shape()[2] >= 3:
            candidates.append((f, img.header.get_data_shape()[2]))

    if not candidates:
        # clean up both temp dirs
        shutil.rmtree(temp_dcm)
        shutil.rmtree(temp_nifti)
        raise RuntimeError("No valid CT NIfTI (≥3 slices) produced")

    best_nifti = max(candidates, key=lambda x: x[1])[0]
    return best_nifti, (temp_nifti, temp_dcm)

def resample_to_cube_sitk(in_nifti: str,
                          side: int = 128) -> str:
    """
    SimpleITK down-resample to EXACT `out_side³` while preserving
    origin / spacing / direction.  Matches the notebook’s
    `resample_image()` quality.

    Returns the new .nii.gz path.
    """
    img = sitk.ReadImage(in_nifti)

    # new isotropic size & spacing
    ref_size     = [side, side, side]
    phys_sz      = [(sz - 1) * sp for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    new_spacing  = [p / (s - 1) for p, s in zip(phys_sz, ref_size)]

    ref_origin    = img.GetOrigin()
    ref_direction = np.identity(3).flatten()

    ref_img = sitk.Image(ref_size, img.GetPixelIDValue())
    ref_img.SetOrigin(ref_origin)
    ref_img.SetSpacing(new_spacing)
    ref_img.SetDirection(ref_direction)

    # center the original inside the reference field-of-view
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(img.GetDirection())
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(
        np.array(img.GetSize()) / 2.0))
    ref_center = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(
        np.array(ref_img.GetSize()) / 2.0))

    centering = sitk.TranslationTransform(3)
    centering.SetOffset(transform.GetInverse().TransformPoint(img_center) - ref_center)
    xform = sitk.CompositeTransform([transform, centering])

    resampled = sitk.Resample(
        img,
        ref_img,
        xform,
        sitk.sitkLinear,
        -1000.0  # default HU outside FOV
    )

    # cast back to int16 (CT)
    resampled = sitk.Cast(resampled, sitk.sitkInt16)

    out_path = in_nifti.replace(".nii", f"_{side}.nii")
    if not out_path.endswith(".gz"):
        out_path += ".gz"
    sitk.WriteImage(resampled, out_path)
    return out_path

def run_dynagan(nifti_file, dynagan_dir,
                alpha_min, alpha_max, alpha_step, gpu_ids):
    """
    nifti_file: the exact 128×128×128 NIfTI you want to feed Dynagan.
    """
    # 1) make a fresh workdir
    workdir = tempfile.mkdtemp(prefix="dynagan_work_")

    # 2) copy your single NIfTI into workdir/imagesTs/
    ts_root = os.path.join(workdir, "imagesTs")
    os.makedirs(ts_root, exist_ok=True)
    dest = os.path.join(ts_root, "LungCT_0000_0000.nii.gz")
    shutil.copy2(nifti_file, dest)

    # 3) compute integer alpha_step (at least 1)
    n_steps = int(alpha_step)

    # 4) build the Dynagan CLI
    cmd = [
        sys.executable,
        os.path.join(dynagan_dir, "test_3D.py"),
        "--dataroot", workdir,
        "--name", "dynagan_tmp",
        "--model", "test",
        "--dataset_mode", "test",
        "--num_test", "1",
        # ---- stop any resizing --------
        "--preprocess", "none",
        "--load_size", "128",
        "--crop_size", "128",
        # -- motion params ------------
        "--alpha_min", str(alpha_min),
        "--alpha_max", str(alpha_max),
        "--alpha_step", str(int(n_steps)),
        "--gpu_ids", gpu_ids,
        "--results_dir", os.path.join(dynagan_dir, "results"),
        "--checkpoints_dir", os.path.join(dynagan_dir, "checkpoints"),
    ]

    # 5) run it _from_ workdir so that "./results" is created there
    subprocess.run(cmd, cwd=dynagan_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 6) pick up the single 4D‐CT file Dynagan writes
    out_file = os.path.join(
        dynagan_dir,
        "results",
        "dynagan_tmp",
        "LungCT_0000_4DCT.nii.gz"
    )
    if not os.path.exists(out_file):
        raise RuntimeError(f"Dynagan did not produce the 4DCT at {out_file}")
    return out_file, workdir


def extract_and_dicomify_torch(nifti4d: str,
                               dvf_dir: str,
                               original_ct_nifti: str,
                               input_dicom_dir: str,
                               output_dir: str):
    """
    • Upsample each Dynagan DVF to native grid with ndimage.zoom
    • Warp ORIGINAL CT through that DVF (sharp) using SpatialTransformer
    • Mutate voxels (+1024 & rotation) and write back as DICOM slices
    """
    # --- 0) load original CT array (Z, Y, X) ---
    orig_img = sitk.ReadImage(original_ct_nifti)
    orig_arr = sitk.GetArrayFromImage(orig_img).astype(np.float32)
    D, H, W  = orig_arr.shape  # depth, height, width

    # --- 1) find all DVF .nii.gz files ---
    dvf_files = sorted(glob.glob(os.path.join(dvf_dir, "*dvf*.nii.gz")))
    if not dvf_files:
        raise RuntimeError(f"No DVF files found in {dvf_dir!r}")

    # --- 2) prepare DICOM headers (sorted by InstanceNumber) once ---
    dicom_fns = sorted(
        glob.glob(os.path.join(input_dicom_dir, "*.dcm")),
        key=lambda fn: int(pydicom.dcmread(fn, stop_before_pixels=True).InstanceNumber)
    )
    headers = [pydicom.dcmread(fn) for fn in dicom_fns]

    # --- 3) we'll init the transformer exactly once ---
    st = None

    for ph, dvf_file in enumerate(dvf_files):
        print(f"Phase {ph:02d}: warping native CT with DVF …")

        # 3a) load the 128³ DVF and upsample to (D,H,W,3)
        nib_dvf   = nib.load(dvf_file)
        dvf_npy   = nib_dvf.get_fdata()                   # e.g. (128,128,128,3)
        # reorder to (Z, Y, X, 3)
        dvf_arr   = np.transpose(dvf_npy, (2, 1, 0, 3))    # → (z128,y128,x128,3)

        # 3b) upsample to native volume (56,512,512,3)
        zoom_facs = (D/dvf_arr.shape[0],
                     H/dvf_arr.shape[1],
                     W/dvf_arr.shape[2],
                     1)
        dvf_arr_up = ndimage.zoom(dvf_arr, zoom=zoom_facs, order=1)

        # 3b) make the flow tensor shape (1, 3, D, H, W)
        dvf_t = (
            torch.from_numpy(dvf_arr_up)
                 .permute(3, 0, 1, 2)   # -> (3, D, H, W)
                 .unsqueeze(0)          # -> (1, 3, D, H, W)
                 .float()
        )
        mov_t = torch.from_numpy(orig_arr)         \
                     .unsqueeze(0).unsqueeze(1)    \
                     .float()                      # -> (1, 1, D, H, W)

        # 3c) init transformer on the DVF’s grid once
        if st is None:
            st = SpatialTransformer(dvf_arr_up.shape[:3])  # pass (D, H, W)
            if torch.cuda.is_available():
                st = st.cuda()

        # 3d) move to GPU if available
        if torch.cuda.is_available():
            dvf_t, mov_t, st_dev = dvf_t.cuda(), mov_t.cuda(), "gpu"
        else:
            st_dev = "cpu"
        warped = st(mov_t, dvf_t).cpu().numpy()[0, 0]  # D,H,W float32

        # --- 4) rotate & offset, then write DICOM slices ---
        # apply your flip/swaps to match ROT, then +1024
        vol = warped[::-1, ::-1, ::-1].transpose((1, 2, 0))  # existing flip+rotate
        vol = vol[:, ::-1, :]  # flip Y
        vol += 1024
        vol = vol.astype(headers[0].pixel_array.dtype)

        phase_dir = os.path.join(output_dir, f"phase_{ph:02d}")
        os.makedirs(phase_dir, exist_ok=True)
        series_uid = pydicom.uid.generate_uid()

        for z in range(D):
            ds = copy.deepcopy(headers[z])
            ds.PixelData         = vol[:, :, z].tobytes()
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID    = pydicom.uid.generate_uid()
            ds.InstanceNumber    = z + 1
            ds.SeriesDescription = f"4DCT_phase_{ph:02d}"
            ds.save_as(os.path.join(phase_dir, f"{z+1:03d}.dcm"))

def main():
    p = argparse.ArgumentParser(
        description="3D→4D CT via Dynagan in one go, back to DICOM"
    )
    p.add_argument("input_dir",
                   help="Folder with raw CT DICOM series")
    p.add_argument("output_dir",
                   help="Where to write the per‐phase DICOM folders")
    p.add_argument("--dynagan_dir", default=r"C:\Users\sebas\Documents\GitHub\TOPAS_ElektaSynergy\ControlHub\src\dynagan",
                   help="Path to your Dynagan repo (where test_3D.py lives)")
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_step", type=float, default=5)
    p.add_argument("--gpu_ids", default="-1",
                   help="GPU IDs for Dynagan; -1 = CPU")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("1) DICOM → NIfTI …")
    nif, (nif_tmpdir, dcm_tmpdir) = find_dicom_series(args.input_dir)

    print("2) Resample to 128³ …")
    nif_rs = resample_to_cube_sitk(nif, side=128)

    print("3) Running Dynagan inference …")
    out4d, workdir = run_dynagan(
    nif_rs,                # ← your 128³ NIfTI path
    args.dynagan_dir,
    args.alpha_min,
    args.alpha_max,
    args.alpha_step,
    args.gpu_ids
)

    print("4) Extracting phases & writing DICOM …")
    extract_and_dicomify_torch(
    nifti4d=out4d,
    dvf_dir=os.path.join(args.dynagan_dir, "results", "dynagan_tmp", "0000", "dvf"),
    original_ct_nifti=nif,
    input_dicom_dir=dcm_tmpdir,
    output_dir=args.output_dir
)

    print("5) Cleaning up temp folders …")
    shutil.rmtree(nif_tmpdir)
    shutil.rmtree(workdir)
    shutil.rmtree(dcm_tmpdir)
    shutil.rmtree(os.path.join(args.dynagan_dir, "results", "dynagan_tmp"))
    shutil.rmtree(os.path.join(args.dynagan_dir, "checkpoints", "dynagan_tmp"))
    print("✅ Done. Your 4D‐CT phases are in:", args.output_dir)
    
if __name__ == "__main__":
    main()
