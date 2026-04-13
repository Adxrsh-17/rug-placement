# Structured Technical Report - Rug Placement Visualizer

## 1. Problem Statement
Design an automated computer-vision pipeline to place rug images realistically on room photos.
The solution should detect valid floor regions, estimate natural placement geometry, and avoid rug overlap with furniture where possible.

## 2. Dataset and Inputs
- Room images: `rug-ai/images/room1.jpeg` to `rug-ai/images/room5.JPG`
- Rug assets: `rug-ai/images/rug1.jpg`, `rug-ai/images/rug3.jpg`
- Segmentation model: SAM ViT-H checkpoint at `rug-ai/models/sam_vit_h_4b8939.pth`

## 3. Pipeline Architecture
The implementation is modularized under `src/rug_visualizer`:

1. Floor detection module (`floor_detection.py`)
- SAM prompt-based floor proposal
- LAB color-statistics filtering
- LBP texture consistency filtering
- Canny-edge boundary exclusion
- Morphological cleanup and contour retention

2. Placement module (`placement.py`)
- Distance-transform center estimation for safe placement
- Perspective trapezoid generation for rug footprint
- Homography-based rug warp onto room plane
- Alpha blending with floor-mask constraint
- Furniture exclusion helpers for difficult scenes

3. End-to-end pipeline (`pipeline.py`)
- Single pair processing
- Room 3 enhanced mode (furniture-aware)
- Full assignment batch processing
- Debug collage and result gallery generation

4. CLI runner (`run_assignment.py`)
- Batch mode, single mode, room3-enhanced mode

## 4. Mathematical Basis
1. Distance transform for placement center:

D(x,y) = min over boundary points of sqrt((x-i)^2 + (y-j)^2)

2. Perspective mapping (homography):

x' = Hx

3. Alpha blending:

I_final = alpha * I_rug + (1 - alpha) * I_room

## 5. Special Handling for Room 3
Room 3 contains strong furniture overlap risk. Enhanced mode uses:
- Edge-density-based candidate furniture detection
- Saturation-based filtering in upper/mid regions
- Bottom-weighted safe-floor clipping
- Placement over safe-floor-only mask

This reduces sofa overlap and improves visual realism.

## 6. Execution Commands
1. Full assignment:
- python run_assignment.py --mode batch

2. One room-rug pair:
- python run_assignment.py --mode single --room room1.jpeg --rug rug1.jpg

3. Enhanced room3:
- python run_assignment.py --mode room3-enhanced --room room3.jpeg --rug rug3.jpg

## 7. Outputs
Generated outputs are stored in `rug-ai/outputs`:
- room*_rug*_result.jpg
- room*_rug*_steps.jpg
- room3_*_enhanced.jpg
- room3_*_enhanced_steps.jpg
- _gallery.jpg

## 8. Repository Readiness
The notebook logic is now represented in reusable script modules and a CLI runner, which is suitable for source control and reproducible execution in a GitHub repository.
