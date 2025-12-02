#!/usr/bin/env python3
"""
DICOM to NIfTI Converter

This script converts DICOM files to NIfTI format using either:
1. dcm2niix command-line tool (preferred, faster)
2. Python libraries (pydicom + nibabel)

Usage:
    python convert_dcm2nii.py <input_dir> <output_dir> [--method auto|dcm2niix|python]
    python convert_dcm2nii.py <input_file.dcm> <output_file.nii.gz> [--method auto|dcm2niix|python]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Union


def check_dcm2niix_available() -> bool:
    """Check if dcm2niix is available in the system."""
    try:
        result = subprocess.run(
            ['dcm2niix', '-h'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_with_dcm2niix(input_path: Path, output_dir: Path, compress: bool = True) -> bool:
    """
    Convert DICOM to NIfTI using dcm2niix command-line tool.
    
    Args:
        input_path: Path to DICOM file or directory
        output_dir: Output directory for NIfTI files
        compress: Whether to compress output (.nii.gz vs .nii)
    
    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'dcm2niix',
        '-z', 'y' if compress else 'n',  # Compress output
        '-f', '%p_%s_%d',  # Filename format: protocol_series_description
        '-o', str(output_dir),  # Output directory
        str(input_path)  # Input path
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Conversion successful!")
            print(result.stdout)
            return True
        else:
            print("✗ Conversion failed!")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out!")
        return False
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False


def convert_with_python(input_path: Path, output_path: Path) -> bool:
    """
    Convert DICOM to NIfTI using Python libraries (pydicom + nibabel).
    
    Args:
        input_path: Path to DICOM file or directory
        output_path: Output NIfTI file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import pydicom
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        print(f"✗ Required libraries not installed: {e}")
        print("Please install: pip install pydicom nibabel")
        return False
    
    try:
        # Handle directory input (series of DICOM files)
        if input_path.is_dir():
            print(f"Reading DICOM series from: {input_path}")
            dcm_files = sorted(input_path.glob('*.dcm'))
            
            if not dcm_files:
                # Try without extension
                dcm_files = [f for f in input_path.iterdir() if f.is_file()]
            
            if not dcm_files:
                print("✗ No DICOM files found in directory!")
                return False
            
            # Read all slices
            slices = []
            for dcm_file in dcm_files:
                try:
                    ds = pydicom.dcmread(str(dcm_file))
                    slices.append(ds)
                except Exception as e:
                    print(f"Warning: Could not read {dcm_file}: {e}")
            
            if not slices:
                print("✗ No valid DICOM files could be read!")
                return False
            
            # Sort slices by instance number or slice location
            slices.sort(key=lambda x: float(getattr(x, 'InstanceNumber', 0)))
            
            # Stack slices into 3D volume
            pixel_arrays = [s.pixel_array for s in slices]
            volume = np.stack(pixel_arrays, axis=-1)
            
            # Get affine transformation from DICOM metadata
            ds = slices[0]
            
        else:
            # Single DICOM file
            print(f"Reading DICOM file: {input_path}")
            ds = pydicom.dcmread(str(input_path))
            volume = ds.pixel_array
            
            # Add dimension if 2D
            if volume.ndim == 2:
                volume = volume[:, :, np.newaxis]
        
        # Extract spacing information
        try:
            pixel_spacing = ds.PixelSpacing  # [row, col]
            slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
            spacing = [float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness]
        except:
            print("Warning: Could not extract spacing information, using default [1, 1, 1]")
            spacing = [1.0, 1.0, 1.0]
        
        # Create affine matrix
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # Try to get orientation from ImageOrientationPatient
        try:
            orientation = ds.ImageOrientationPatient
            position = ds.ImagePositionPatient
            
            # Create rotation matrix from orientation
            row_cosine = np.array(orientation[:3])
            col_cosine = np.array(orientation[3:])
            slice_cosine = np.cross(row_cosine, col_cosine)
            
            affine[0, :3] = row_cosine * spacing[0]
            affine[1, :3] = col_cosine * spacing[1]
            affine[2, :3] = slice_cosine * spacing[2]
            affine[:3, 3] = position
        except:
            print("Warning: Could not extract orientation information")
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(volume, affine)
        
        # Save NIfTI file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nii_img, str(output_path))
        
        print(f"✓ Conversion successful!")
        print(f"  Output: {output_path}")
        print(f"  Shape: {volume.shape}")
        print(f"  Spacing: {spacing}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert DICOM files to NIfTI format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a directory of DICOM files
  python convert_dcm2nii.py /path/to/dicom/dir /path/to/output/dir
  
  # Convert a single DICOM file
  python convert_dcm2nii.py input.dcm output.nii.gz
  
  # Force using Python method
  python convert_dcm2nii.py /path/to/dicom/dir /path/to/output/dir --method python
        """
    )
    
    parser.add_argument('input', type=str, help='Input DICOM file or directory')
    parser.add_argument('output', type=str, help='Output NIfTI file or directory')
    parser.add_argument(
        '--method',
        choices=['auto', 'dcm2niix', 'python'],
        default='auto',
        help='Conversion method (default: auto)'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress output (save as .nii instead of .nii.gz)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input
    if not input_path.exists():
        print(f"✗ Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Determine conversion method
    if args.method == 'auto':
        if check_dcm2niix_available():
            method = 'dcm2niix'
            print("Using dcm2niix (command-line tool)")
        else:
            method = 'python'
            print("dcm2niix not found, using Python libraries")
    else:
        method = args.method
        print(f"Using {method} method")
    
    # Perform conversion
    success = False
    
    if method == 'dcm2niix':
        if not check_dcm2niix_available():
            print("✗ dcm2niix is not available!")
            print("Install it with: sudo apt-get install dcm2niix")
            print("Or use --method python")
            sys.exit(1)
        
        # For dcm2niix, output should be a directory
        if output_path.suffix in ['.nii', '.gz']:
            output_dir = output_path.parent
        else:
            output_dir = output_path
        
        success = convert_with_dcm2niix(input_path, output_dir, compress=not args.no_compress)
    
    elif method == 'python':
        # For Python method, ensure output has proper extension
        if output_path.is_dir() or not output_path.suffix:
            if input_path.is_dir():
                output_file = output_path / 'output.nii.gz'
            else:
                output_file = output_path / (input_path.stem + '.nii.gz')
        else:
            output_file = output_path
        
        # Ensure proper extension
        if not args.no_compress and not str(output_file).endswith('.nii.gz'):
            output_file = output_file.with_suffix('.nii.gz')
        elif args.no_compress and output_file.suffix != '.nii':
            output_file = output_file.with_suffix('.nii')
        
        success = convert_with_python(input_path, output_file)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
