#!/usr/bin/env python3
"""
Batch DICOM to NIfTI Converter for ADNI PET Dataset

This script processes all patient directories in the ADNI dataset and converts
DICOM files to NIfTI format.
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm


def find_dicom_directories(base_dir):
    """
    Find all directories containing DICOM files in the ADNI dataset.
    
    Returns:
        List of tuples: (patient_id, dicom_dir_path)
    """
    base_path = Path(base_dir)
    dicom_dirs = []
    
    # Iterate through patient directories
    for patient_dir in sorted(base_path.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        # Look for AV45 PET scan directory
        av45_dir = patient_dir / "AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution"
        
        if av45_dir.exists():
            # Find the deepest directory containing .dcm files
            for date_dir in av45_dir.iterdir():
                if date_dir.is_dir():
                    for scan_dir in date_dir.iterdir():
                        if scan_dir.is_dir():
                            # Check if this directory contains .dcm files
                            dcm_files = list(scan_dir.glob("*.dcm"))
                            if dcm_files:
                                dicom_dirs.append((patient_id, scan_dir))
                                break
    
    return dicom_dirs


def convert_patient(patient_id, dicom_dir, output_base_dir, no_compress=True):
    """
    Convert a single patient's DICOM files to NIfTI.
    
    Args:
        patient_id: Patient identifier
        dicom_dir: Path to directory containing DICOM files
        output_base_dir: Base directory for output files
        no_compress: If True, save as .nii; if False, save as .nii.gz
    
    Returns:
        True if successful, False otherwise
    """
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    extension = ".nii" if no_compress else ".nii.gz"
    output_file = output_dir / f"{patient_id}_AV45{extension}"
    
    # Skip if already exists
    if output_file.exists():
        print(f"  ⊙ Skipping {patient_id} (already exists)")
        return True
    
    # Build command
    cmd = [
        "python",
        "convert_dcm2nii.py",
        str(dicom_dir),
        str(output_file)
    ]
    
    if no_compress:
        cmd.append("--no-compress")
    
    # Run conversion
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"  ✓ {patient_id}")
            return True
        else:
            print(f"  ✗ {patient_id} - Conversion failed")
            print(f"    Error: {result.stderr[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ {patient_id} - Timeout")
        return False
    except Exception as e:
        print(f"  ✗ {patient_id} - Error: {e}")
        return False


def main():
    # Configuration
    base_dir = "/home/prml/RIMA/datasets/ADNI/AD_PET_130_FIN/ADNI"
    output_dir = "/home/prml/RIMA/datasets/ADNI/AD_PET_130_FIN/ADNI_NII"
    
    print("=" * 70)
    print("ADNI PET DICOM to NIfTI Batch Converter")
    print("=" * 70)
    print(f"Input directory:  {base_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Find all DICOM directories
    print("\nScanning for DICOM files...")
    dicom_dirs = find_dicom_directories(base_dir)
    
    print(f"Found {len(dicom_dirs)} patients with DICOM data\n")
    
    if not dicom_dirs:
        print("No DICOM directories found!")
        return
    
    # Process each patient
    print("Starting conversion...\n")
    
    successful = 0
    failed = 0
    
    for patient_id, dicom_dir in tqdm(dicom_dirs, desc="Converting"):
        success = convert_patient(patient_id, dicom_dir, output_dir, no_compress=True)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total patients:  {len(dicom_dirs)}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {failed}")
    print(f"Success rate:    {successful/len(dicom_dirs)*100:.1f}%")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
