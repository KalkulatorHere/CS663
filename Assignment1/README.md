# CS663: Digital Image Processing - Assignment 1

This repository contains the code and data for **CS663 Assignment 1** (Digital Image Processing), as per the assignment PDF. The assignment covers three main topics: **Image Subsampling & Interpolation**, **Thresholding**, and **Contrast Enhancement**. Each topic is implemented in a dedicated subfolder, with code organized by question and subpart.

> **Note**: This report contains all outputs including processed images, plots, and analysis results for comprehensive evaluation.
Note: Markdown written with Claude

---

## Folder Structure

```
Assignment1/
│
├── assignment_1_InterpThreshHist.pdf   # Assignment description
│
├── data/                              # All input images and .mat files
│   ├── hist/
│   ├── interp/
│   └── thresh/
│
├── Question1/                         # Image Subsampling & Interpolation
│   ├── BicubicInterpolation.py
│   ├── BilinearInterpolation.py
│   ├── CTInterpolationComparison.py
│   ├── ImageRotation.py
│   ├── ImageShrink.py
│   ├── ImageUtils.py
│   ├── NearestNeighborInterpolation.py
│   ├── main.py
│   └── main_but_jupytercell.ipynb
│
├── Question2/                         # Thresholding
│   ├── AdaptiveThresholder.py
│   ├── ImageDisplayer.py
│   ├── ImageLoader.py
│   ├── ManualThresholder.py
│   ├── OtsuThresholder.py
│   ├── main.py
│   └── main_but_jupytercell.ipynb
│
├── Question3/                         # Contrast Enhancement
│   ├── CLAHE.py
│   ├── HistogramEqualization.py
│   ├── HistogramMatching.py
│   ├── LinearContrastStretch.py
│   ├── main.py
│   └── main_but_jupytercell.ipynb
│
└── (other output and cache folders)
```

---

## Assignment Overview

### **1. Image Subsampling and Interpolation** (`Question1/`)
- **ImageShrink.py**: Implements `myImageShrink()` for image downsampling (Q1a).
- **NearestNeighborInterpolation.py**: Implements `myNearestNeighborInterpolation()` for nearest-neighbor upsampling (Q1b).
- **BilinearInterpolation.py**: Implements `myBilinearInterpolation()` for bilinear upsampling (Q1c).
- **BicubicInterpolation.py**: Implements `myBicubicInterpolation()` for bicubic upsampling (Q1d).
- **ImageRotation.py**: Implements image rotation using bilinear and nearest-neighbor interpolation (Q1e).
- **CTInterpolationComparison.py**: Compares interpolation methods on CT images, computes RMSE, and visualizes results (Q1f).
- **ImageUtils.py**: Utility functions for image loading, displaying, and processing.
- **main.py / main_but_jupytercell.ipynb**: Entry points to run and visualize results for all subparts.

### **2. Thresholding** (`Question2/`)
- **ManualThresholder.py**: Implements manual thresholding (Q2a).
- **OtsuThresholder.py**: Implements Otsu's thresholding (Q2b).
- **AdaptiveThresholder.py**: Implements local/adaptive thresholding (Q2c).
- **ImageLoader.py**: Loads images for thresholding.
- **ImageDisplayer.py**: Handles image display and visualization.
- **main.py / main_but_jupytercell.ipynb**: Entry points to run and visualize thresholding results.

### **3. Contrast Enhancement** (`Question3/`)
- **LinearContrastStretch.py**: Implements `myLinearContrastStretch()` for linear contrast stretching (Q3a).
- **HistogramEqualization.py**: Implements `myHistEqualize()` for histogram equalization (Q3b).
- **CLAHE.py**: Implements `myCLAHE()` for contrast-limited adaptive histogram equalization (Q3c).
- **HistogramMatching.py**: Implements `myHistMatch()` for histogram matching (Q3d).
- **utils.py**: Utility functions for histogram computation and image processing.
- **main.py / main_but_jupytercell.ipynb**: Entry points to run and visualize contrast enhancement results.

---

## Data

All input images are organized under `data/` with subfolders:
- `hist/` for histogram/contrast enhancement images
- `interp/` for interpolation and rotation images
- `thresh/` for thresholding images

---

##  How Each Class/Script Maps to Assignment Questions

Each Python file implements a specific function or algorithm as required by the assignment subparts. The main scripts (`main.py` or notebooks) in each question folder demonstrate usage, visualization, and parameter tuning as per the assignment. Utility scripts (e.g., `ImageUtils.py`, `utils.py`) provide shared helper functions for image I/O, display, and processing.

---

##  How to Run

1. **Install dependencies** (if any, e.g., `numpy`, `matplotlib`, `opencv-python`, `scipy`).
2. **Navigate** to the relevant question folder.
3. **Run** the main script or open the Jupyter notebook:
   ```sh
   python main.py
   ```
   or
   ```sh
   jupyter notebook main_but_jupytercell.ipynb
   ```
4. **Outputs** (images, plots) are displayed or saved in the respective output folders.

---

##  Notes

- **Colormaps**: All images are displayed with at least 200 colors/intensities, and colorbars are shown as required.
- **Floating-point images**: Unless visualizing RGB, images are kept in floating-point format.
- **Parameter tuning**: For adaptive/CLAHE/histogram matching, parameters are manually tuned as per assignment instructions.
- **Output**: Processed images and results are saved in `output/` or `output_thresholding/` folders.

---

##  Reference

See `assignment_1_InterpThreshHist.pdf` for detailed problem statements and requirements.

---

**For any clarifications, refer to the assignment PDF or the code comments in each script.**
