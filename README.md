# OT-based-Heterogeneous-Multi-Modal-Fusion-Embedding-for-AD-analysis

This repository provides an Optimal Transport (OT) based framework for heterogeneous multi-modal fusion embedding, specifically designed for **Alzheimer's Disease (AD) analysis**. 

The core methodology is built upon [RIMA](https://github.com/Qinkaiyu/RIMA) and has been significantly modified to accommodate and fuse heterogeneous medical imaging modalities.



## Key Features
* **Heterogeneous Multi-modal Fusion**: Specialized in integrating diverse data sources for AD diagnostic analysis.
* **OT-based Embedding**: Leverages Optimal Transport to align and fuse different feature distributions effectively.
* **ADNI Data Support**: Pre-configured to process and analyze **MRI** and **PET** imaging data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).
* **Optimization in Progress**: Currently transitioning to **JAX** dependencies to achieve significantly faster training speeds and improved computational efficiency.

## Data Modalities
* **MRI (Magnetic Resonance Imaging)**: Structural brain data.
* **PET (Positron Emission Tomography)**: Functional/metabolic brain data.
* *Dataset source: ADNI (Alzheimer's Disease Neuroimaging Initiative)*

## Acknowledgments
This project is a modified version of the original work found in:
* [Qinkaiyu/RIMA](https://github.com/Qinkaiyu/RIMA) - *Robust Inter-modal Alignment for Multi-modal Learning.*

## Current Status & Roadmap
- [x] Initial modification for heterogeneous AD data.
- [x] Integration of MRI and PET data processing.
- [ ] **Fixing JAX dependencies** for high-performance training (In-progress).
