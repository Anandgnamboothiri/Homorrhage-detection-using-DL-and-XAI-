# Brain Hemorrhage Detection (Binary Classification)

Binary classification of brain CT scans: **Hemorrhagic** vs **NORMAL** using JPG images and transfer learning (ResNet50).

## Dataset Structure

Place your images in the following structure:

```
Data/
    Hemorrhagic/          # Label 1 - hemorrhage present
        (subfolders with .jpg images)
    NORMAL/               # Label 0 - no hemorrhage
        (subfolders with .jpg images)
```

## Setup

```bash
pip install -r requirements.txt
```

Or use conda:
```bash
conda env create -f environment.yml
# Brain Hemorrhage Detection

Binary classification of brain CT scans (JPG/PNG) to detect intracranial hemorrhage vs normal.

This repository provides training code, inference utilities, a small Flask demo, and Grad‑CAM visualization to help interpret model decisions.

**Key points**
- Model: ResNet50 backbone (transfer learning)
- Input size: 224×224×3
- Binary output: probability of hemorrhage (sigmoid)
- Demo: Flask app with Grad‑CAM visualization

## Quick Start

Prerequisites (example): Python 3.8+ and required packages.

Install dependencies:

```bash
pip install -r requirements.txt
```

Or create the conda environment:

```bash
conda env create -f environment.yml
conda activate brainenv
```

## Dataset layout

Organize your dataset like this:

```
Data/
    NORMAL/          # class 0 - non-hemorrhagic images
        ...
    Hemorrhagic/     # class 1 - hemorrhagic images
        ...
```

Each class folder may contain images directly or in subfolders — Keras `flow_from_directory` will recurse.

## Training

Train using the provided script:

```bash
python train.py
```

Notes:
- `train.py` expects `DATASET_PATH` to point to your `Data/` directory (default is the project Data path).
- Training uses an 80/20 train/validation split and basic augmentation.
- The trained model is saved to `ichdemo/trained_model.h5` by default.

## Inference & Demo

Start the simple web demo (Flask):

```bash
cd ichdemo
python app.py
```

Open http://localhost:9999/ in your browser and upload a brain CT image (JPG/PNG). The app shows the prediction probability and a Grad‑CAM heatmap if available.

Files of interest:
- `train.py` — training pipeline and model definition
- `ichdemo/magicsauce.py` — preprocessing and model path
- `ichdemo/gradcam.py` — Grad‑CAM utilities
- `ichdemo/app.py` — Flask demo and upload handler

## Usage examples

- Train locally (single machine / CPU or GPU):

```bash
python train.py
```

- Run demo after placing or training a model:

```bash
cd ichdemo
python app.py
```

## Implementation details
- Backbone: `keras.applications.ResNet50` (ImageNet weights)
- Top: GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)
- Loss: `binary_crossentropy` • Optimizer: Adam (1e-5)
- Input preprocessing: resize to 224×224, normalize to [0,1]

## Notes & troubleshooting
- If the Flask app prints "Model not found", run `train.py` or copy a trained `trained_model.h5` into `ichdemo/`.
- Grad‑CAM generation may fail for unexpected model layer names; `gradcam.py` contains heuristics and verbose logs to help debug.

## Project structure

- `train.py` — training script
- `ichdemo/` — demo app, preprocessing and visualization helpers
    - `app.py` — Flask app
    - `magicsauce.py` — preprocessing and constants
    - `gradcam.py` — Grad‑CAM utilities
    - `trained_model.h5` — (optional) pre-trained model
- `Data/` — expected dataset directory (not committed)
- `requirements.txt`, `environment.yml` — environment specs

## License & attribution
This repository is provided as-is for educational purposes. Add your preferred license if you plan to share the project publicly.

----

Updated README for this project. See `ichdemo/app.py` and `train.py` for runnable examples.

project pre trained model: https://drive.google.com/file/d/1eX5pK-zKbfqc6gQZeAsFIBaDYMfPe0ME/view?usp=sharing

