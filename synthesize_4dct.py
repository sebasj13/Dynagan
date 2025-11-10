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
try:
    from .util.spatialTransform import SpatialTransformer
except ImportError:
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
            dicom_intercept = float(ds.RescaleIntercept)
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
        reorient=False     # for v2.6.0
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
    return best_nifti, (temp_nifti, temp_dcm), dicom_intercept

def resample_to_cube_sitk(in_nifti: str, side: int = 128) -> str:
    img = sitk.ReadImage(in_nifti)

    # target size and spacing that preserves physical extent
    ref_size = [side, side, side]
    phys_sz  = [(sz - 1) * sp for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    new_spacing = [p / (s - 1) for p, s in zip(phys_sz, ref_size)]

    # build reference image that *copies* original orientation & origin
    ref_img = sitk.Image(ref_size, img.GetPixelIDValue())
    ref_img.SetOrigin(img.GetOrigin())
    ref_img.SetSpacing(new_spacing)
    ref_img.SetDirection(img.GetDirection())  # <- keep orientation!

    # identity transform (no extra rotations/translations)
    xform = sitk.Transform(3, sitk.sitkIdentity)

    resampled = sitk.Resample(
        img,
        ref_img,
        xform,
        sitk.sitkLinear,
        -1000.0
    )

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

    return workdir


# ---- helper: nibabel-style DVF upsample to target shape (D,H,W) ----
def _load_and_resize_dvf_nib(dvf_path: str, target_shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """
    Read DVF NIfTI (expected (X,Y,Z,3)), return numpy as (D,H,W,3) with target_shape_zyx via ndimage.zoom.
    Uses order=1 (linear) and does NOT scale displacement magnitudes.
    """
    nii = nib.load(dvf_path)
    data = nii.get_fdata(dtype=np.float32)  # (X,Y,Z,3)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"DVF must have shape (X,Y,Z,3); got {data.shape}")

    # Reorder to (Z,Y,X,3) to match SimpleITK array orientation (orig_arr is (Z,Y,X))
    dvf_zyx3 = np.transpose(data, (2, 1, 0, 3)).astype(np.float32)  # (Z,Y,X,3)

    Dz, Hy, Wx = target_shape_zyx
    z0, y0, x0, _ = dvf_zyx3.shape

    # Zoom factors per axis to hit the CT voxel grid size
    zf = Dz / z0
    yf = Hy / y0
    xf = Wx / x0

    # Linear interpolation, no smoothing
    dvf_up = ndimage.zoom(dvf_zyx3, zoom=(zf, yf, xf, 1), order=1)
    dvf_up = dvf_up.astype(np.float32, copy=False)  # (D,H,W,3)
    return dvf_up

# ---- optional: write vector DVF as NIfTI using CT geometry (affine/header) ----
def _save_dvf_as_nifti_with_ct_geom(dvf_zyx3: np.ndarray, ct_ref_path: str, out_path: str):
    """
    Save DVF (Z,Y,X,3) as NIfTI using CT’s affine+header so spacing/origin/direction are correct.
    NIfTI data will be stored as (X,Y,Z,3).
    """
    if dvf_zyx3.ndim != 4 or dvf_zyx3.shape[-1] != 3:
        raise ValueError(f"Expected (Z,Y,X,3), got {dvf_zyx3.shape}")

    # Back to nibabel orientation (X,Y,Z,3)
    dvf_xyzc = np.transpose(dvf_zyx3, (2, 1, 0, 3))

    ref = nib.load(ct_ref_path)
    hdr = ref.header.copy()
    aff = ref.affine.copy()

    # Ensure float32
    dvf_xyzc = dvf_xyzc.astype(np.float32, copy=False)
    nib.save(nib.Nifti1Image(dvf_xyzc, aff, hdr), out_path)


def extract_and_dicomify_torch(
    dvf_dir: str,
    original_ct_nifti: str,
    input_dicom_dir: str,
    output_dir: str,
    intercept: float = 0.0,
    callback=None
):
    """
    • Upsample each Dynagan DVF to native CT grid with nibabel + ndimage.zoom (like the README flow)
    • Warp ORIGINAL CT via SpatialTransformer using that upsampled DVF
    • Write out: resampled DVFs (NIfTI) + warped phases (NIfTI)
    """
    # --- 0) load original CT array (Z, Y, X) ---
    orig_img = sitk.ReadImage(original_ct_nifti)
    orig_arr = sitk.GetArrayFromImage(orig_img).astype(np.float32)
    D, H, W = orig_arr.shape  # depth, height, width (SimpleITK order)

    # --- 1) DVFs to process ---
    dvf_files = sorted(glob.glob(os.path.join(dvf_dir, "*dvf*.nii*")))
    if not dvf_files:
        raise RuntimeError(f"No DVF files found in {dvf_dir!r}")

    # --- 2) prepare DICOM headers (sorted by InstanceNumber) once (kept as in your code) ---
    dicom_fns = sorted(
        glob.glob(os.path.join(input_dicom_dir, "*.dcm")),
        key=lambda fn: int(pydicom.dcmread(fn, stop_before_pixels=True).InstanceNumber)
    )
    headers = [pydicom.dcmread(fn) for fn in dicom_fns]  # not used here, kept for parity

    # --- 3) spatial transformer (initialized once on native grid) ---
    st = None
    os.makedirs(os.path.join(output_dir, "DVFs"), exist_ok=True)

    for ph, dvf_file in enumerate(dvf_files):
        if callback is not None:
            # progress: 3 setup steps + phases; keep your original style
            callback((3 + ph + 1) / (4 + len(dvf_files)))

        # --- A) Upsample DVF to native CT size using nibabel + ndimage.zoom ---
        dvf_arr_up = _load_and_resize_dvf_nib(dvf_file, (D, H, W))  # (D,H,W,3)
       
        # If your DVF was in voxels of the 128³ grid (not mm), preserve physical motion by scaling:
        zf, yf, xf = D/128.0, H/128.0, W/128.0
        dvf_arr_up[..., 0] *= xf
        dvf_arr_up[..., 1] *= yf
        dvf_arr_up[..., 2] *= zf

        # --- B) Save DVF (native grid) as NIfTI with CT geometry (optional but useful) ---
        # also keep original for traceability
        shutil.copy2(dvf_file, os.path.join(output_dir, "DVFs", os.path.basename(dvf_file)))
        dvf_native_path = os.path.join(output_dir, "DVFs", f"dvf_phase_{ph:02d}_native.nii.gz")
        #_save_dvf_as_nifti_with_ct_geom(dvf_arr_up, original_ct_nifti, dvf_native_path)

        # --- C) Torch tensors for warping (same as your workflow) ---
        dvf_t = (
            torch.from_numpy(dvf_arr_up[..., [2, 1, 0]])  # (D,H,W,3)
                 .permute(3, 0, 1, 2)     # -> (3, D, H, W)
                 .unsqueeze(0)            # -> (1, 3, D, H, W)
                 .float()
        )

        mov_t = (
            torch.from_numpy(orig_arr)    # (D,H,W)
                 .unsqueeze(0).unsqueeze(1)  # -> (1, 1, D, H, W)
                 .float()
        )

        # init transformer on native CT grid once
        if st is None:
            st = SpatialTransformer(np.asarray([D, H, W]))
            if torch.cuda.is_available():
                st = st.cuda()

        # move to GPU if available
        if torch.cuda.is_available():
            dvf_t = dvf_t.cuda()
            mov_t = mov_t.cuda()

        warped = st(mov_t, dvf_t).cpu().numpy()[0, 0]  # (D,H,W), float32

        # optional intercept shift (mirrors your function signature)
        if intercept != 0.0:
            warped = warped + float(intercept)

        vol = warped[::-1, ::-1, ::-1].transpose((1, 2, 0))  # existing flip+rotate
        vol = np.rot90(vol, axes=(0,1), k=2) 
        vol = vol.astype(headers[0].pixel_array.dtype)
        vol = vol[:, :, ::-1]

        phase_dir = os.path.join(output_dir, f"phase_{ph:02d}")
        os.makedirs(phase_dir, exist_ok=True)
        series_uid = pydicom.uid.generate_uid()
        shutil.copy2(dvf_file, os.path.join(os.path.join(output_dir, "DVFs"), os.path.basename(dvf_file)))
        for z in range(len(headers)):
            ds = copy.deepcopy(headers[z])
            ds.PixelData         = vol[:, :, z].tobytes()
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID    = pydicom.uid.generate_uid()
            ds.InstanceNumber    = z + 1
            ds.SeriesDescription = f"4DCT_phase_{ph:02d}"
            ds.RescaleIntercept  = 0.0
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
    nif, (nif_tmpdir, dcm_tmpdir), intercept = find_dicom_series(args.input_dir)

    print("2) Resample to 128³ …")
    nif_rs = resample_to_cube_sitk(nif, side=128)

    print("3) Running Dynagan inference …")
    workdir = run_dynagan(
    nif_rs,                # ← your 128³ NIfTI path
    args.dynagan_dir,
    args.alpha_min,
    args.alpha_max,
    args.alpha_step,
    args.gpu_ids
)

    print("4) Extracting phases & writing DICOM …")
    extract_and_dicomify_torch(
    dvf_dir=os.path.join(args.dynagan_dir, "results", "dynagan_tmp", "0000", "dvf"),
    original_ct_nifti=nif,
    input_dicom_dir=dcm_tmpdir,
    output_dir=args.output_dir,
    intercept=0
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
