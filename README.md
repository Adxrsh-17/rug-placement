# Advanced Rug Placement Visualizer

Structured internship assignment for automatic rug placement on room images using computer vision and SAM segmentation.

## Highlights

- Detects floor region with SAM + refinement pipeline
- Finds stable rug placement using distance transform
- Applies perspective warp for realistic orientation
- Blends rug with alpha mask for seamless composition
- Includes enhanced Room 3 workflow with furniture exclusion

## Tech Stack

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Segment Anything Model (SAM)

## Project Layout

```text
VisualizerAssignment/
├─ run_assignment.py
├─ requirements.txt
├─ rug_visualizer.ipynb
├─ src/
│  └─ rug_visualizer/
│     ├─ __init__.py
│     ├─ floor_detection.py
│     ├─ placement.py
│     └─ pipeline.py
├─ rug-ai/
│  ├─ images/
│  ├─ models/
│  └─ outputs/
└─ REPORT/
   ├─ Report.pdf
   └─ structured_report.md
```

## Setup

1. Create and activate virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download SAM checkpoint and place it in rug-ai/models/

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P rug-ai/models/
```

## Run

Run complete assignment batch:

```bash
python run_assignment.py --mode batch
```

Run single room and rug:

```bash
python run_assignment.py --mode single --room room1.jpeg --rug rug1.jpg
```

Run enhanced Room 3 pipeline:

```bash
python run_assignment.py --mode room3-enhanced --room room3.jpeg --rug rug3.jpg
```

Disable debug collages:

```bash
python run_assignment.py --mode batch --no-debug
```

## Outputs

Generated files are written to rug-ai/outputs/:

- room*_rug*_result.jpg
- room*_rug*_steps.jpg
- room3_*_enhanced.jpg
- room3_*_enhanced_steps.jpg
- _gallery.jpg

## Mathematical Formulation

Distance transform objective:

$$
D(x,y) = \min_{(i,j) \in \partial\Omega} \sqrt{(x-i)^2 + (y-j)^2}
$$

Homography mapping:

$$
\mathbf{x'} = H\mathbf{x}
$$

Alpha compositing:

$$
I_{final}(p) = \alpha(p)I_{rug}(p) + (1-\alpha(p))I_{room}(p)
$$

## Notes

- GPU improves SAM performance significantly, CPU is supported.
- Large images are downsampled internally for stability.
- Notebook is kept as reference; script workflow is recommended for repo use.

