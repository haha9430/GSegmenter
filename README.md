# GSegmenter 🛰️

> **Intelligent 3D Object Segmentation, Gaussian Grouping, and Interactive Scene Editing with SAM + 3D Gaussian Splatting**

GSegmenter는 **3D Gaussian Splatting**(3DGS)의 고속 렌더링 성능과 **Segment Anything Model**(SAM)의 범용 분할 능력을 결합하여, 정적인 3D 장면을 객체 단위로 이해하고 편집할 수 있도록 설계된 시스템입니다.

기존 3DGS는 매우 높은 품질의 장면 재구성과 실시간 렌더링을 제공하지만, 객체 수준의 분할과 조작에는 직접적으로 대응하지 못합니다.  
GSegmenter는 이 한계를 보완하기 위해 멀티뷰 이미지에서 SAM이 예측한 2D segmentation mask를 3D Gaussian 공간으로 lifting하고, **Gaussian Grouping**을 통해 장면 내 개별 객체 또는 영역을 일관된 그룹으로 구성합니다.

이를 통해 사용자는 특정 객체를 선택하고, 이동, 회전, 삭제, 재배치와 같은 **interactive editing** 작업을 수행할 수 있습니다.

---

## ✨ Key Features

- **Semantic Lifting from 2D to 3D**  
  SAM이 생성한 2D segmentation mask를 여러 시점에서 수집하고, 이를 3D Gaussian에 역투영하여 객체별 후보 identity를 형성합니다.

- **Gaussian Grouping**  
  각 Gaussian에 identity-related representation을 부여하여 instance 또는 stuff membership을 학습하고, 멀티뷰에서 일관된 객체 그룹을 형성합니다.

- **Multi-view Identity Consistency**  
  단일 이미지의 noisy mask에 의존하지 않고, 여러 시점의 투표와 공간적 일관성 제약을 통해 안정적인 3D object grouping을 수행합니다.

- **Interactive Object Manipulation**  
  선택된 Gaussian group에 대해 translation, rotation, deletion 등의 객체 단위 변환을 적용할 수 있습니다.

- **Local Gaussian Editing**  
  그룹화된 Gaussian을 기반으로 object removal, local recomposition, appearance-level editing 같은 후처리를 효율적으로 수행할 수 있습니다.

- **Geometric Boundary Infilling**  
  객체 이동 또는 삭제 이후 생기는 빈 공간과 경계 불연속성을 주변 Gaussian 분포를 활용해 완화하도록 설계됩니다.

- **Real-time Rendering**  
  3DGS의 장점을 유지해 편집 중에도 높은 해상도와 실시간 수준의 응답성을 목표로 합니다.

---

## 🧠 Why Gaussian Grouping

Gaussian Grouping은 3DGS에 객체 수준 의미를 부여하는 핵심 개념입니다.

기존의 Gaussian은 주로 장면의 geometry와 appearance를 표현하는 데 집중하지만, Gaussian Grouping에서는 각 Gaussian이 **object identity** 또는 **group membership**에 대한 표현도 함께 가집니다.  
이 representation은 SAM의 2D mask supervision과 멀티뷰 consistency를 통해 학습되며, 서로 다른 시점에서 관측된 동일 객체를 하나의 coherent group으로 정렬하는 역할을 수행합니다.

이 접근은 다음과 같은 장점을 가집니다:

- **3D annotation 없이 object-level grouping 가능**
- **Open-world scene에 대한 유연한 segmentation**
- **학습 후 grouped Gaussian을 바로 편집 단위로 활용 가능**
- **removal, recoloring, recomposition 등 downstream editing으로 확장 가능**

---

## 🛠️ System Pipeline

### 1. Data Acquisition
입력 이미지는 **COLMAP**을 통해 카메라 포즈와 sparse point cloud로 정렬되며, 이는 이후 Gaussian Splatting 학습의 기하적 초기 조건으로 사용됩니다.

### 2. 3D Gaussian Splatting Training
NerfStudio 기반 파이프라인 또는 유사한 3DGS 학습 절차를 통해 장면을 구성하는 Gaussian의 위치, 크기, 공분산, 색상, opacity 등을 최적화합니다.

### 3. SAM-based Mask Extraction
각 입력 뷰에 대해 **SAM**을 적용하여 객체 및 배경 영역에 대한 2D segmentation mask를 생성합니다.

### 4. Identity Lifting and Gaussian Grouping
생성된 2D mask는 카메라 파라미터를 이용해 3D Gaussian 공간으로 대응되며, Gaussian별 identity encoding 또는 object membership score를 누적합니다.  
이후 멀티뷰 voting과 spatial consistency를 이용해 동일 객체에 속하는 Gaussian들이 하나의 coherent group으로 정리됩니다.

### 5. Interactive Editing
그룹화된 Gaussian은 편집 가능한 scene entity로 취급되며, 사용자는 특정 group을 선택하여 이동, 회전, 삭제, 분리, 재배치 등을 수행할 수 있습니다.

### 6. Geometric Infilling and Scene Repair
객체 제거 또는 이동 이후 발생하는 장면 불연속성은 주변 Gaussian 밀도와 경계 정보를 활용해 자연스럽게 보정하는 방향으로 처리됩니다.

---

## 🏗️ System Architecture

| Module                 | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| **Data Acquisition**   | COLMAP 기반 포즈 추정과 sparse reconstruction 수행                             |
| **GS Training**        | NerfStudio/3DGS 기반 장면 재구성 및 Gaussian 파라미터 최적화                   |
| **Gaussian Grouping**  | SAM 마스크를 3D Gaussian identity로 lifting하고 멀티뷰 일관성을 통해 그룹 형성 |
| **Interactive Editor** | Gaussian group 단위 selection, transform, deletion, repair 수행                |

---

## 🎯 Editing Scenarios

GSegmenter는 다음과 같은 객체 중심 편집 시나리오를 지원하는 것을 목표로 합니다.

- **Object Selection**  
  장면에서 특정 객체를 클릭하거나 선택해 해당 Gaussian group을 활성화합니다.

- **Object Translation / Rotation**  
  선택된 group에 transformation matrix를 적용해 위치와 방향을 바꿉니다.

- **Object Removal**  
  선택 객체를 scene representation에서 제거해 장면을 정리합니다.

- **Scene Recomposition**  
  여러 Gaussian group을 재배치해 장면 구성을 바꿉니다.

- **Appearance Editing**  
  grouped representation을 활용해 색상 변경이나 style-aware editing으로 확장할 수 있습니다.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA 11.8 or 12.1
- NVIDIA GPU (RTX 30/40 Series recommended)

### Installation

```bash
# After cloning the repository, work from the repo root.
cd GSegmenter

conda create -n gsegmenter python=3.10 -y
conda activate gsegmenter

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nerfstudio
pip install -r requirements.txt
```

For the recommended multi-environment setup, keep SAM 2 in a separate virtual
environment so its PyTorch version does not conflict with the gsplat /
NerfStudio environment. Install the SAM 2 environment with
`requirements-sam2.txt`, then install the official SAM 2 package inside that
environment.

### Project Layout

The first scaffold follows a NerfStudio-first split:

- `configs/`: Hydra-style config presets for dataset, training, mapping, and editor runs.
- `gsegmenter/data/`: COLMAP and scene-loading code.
- `gsegmenter/training/`: NerfStudio / 3DGS training adapters.
- `gsegmenter/segmentation/`: SAM and other 2D mask extractors.
- `gsegmenter/mapping/`: 2D-to-3D identity lifting and voting.
- `gsegmenter/editor/`: interactive transforms and scene repair.
- `gsegmenter/render/`: projection and rendering helpers.
- `gsegmenter/utils/`: shared math, logging, and coordinate-frame utilities.
- `scripts/`: runnable entry points.
- `tools/`: utility scripts for mask extraction and grouping.
- `notebooks/`: research-only analysis and visualization.

The first runnable training wrapper is `scripts/train_splatfacto.py`, which
constructs a NerfStudio `ns-train splatfacto` invocation from the project
config.
If `ns-train` is not on `PATH`, pass `--ns-train-bin <full-path-or-command>` to
the wrapper. If your dataset is not stored under `data/scene01`, pass
`--data-path <prepared-nerfstudio-dataset-root>` instead.
For real captured videos, use `scripts/prepare_video_scene.py` first. It wraps
`ns-process-data video` so a single `.mp4` can be converted into a
NerfStudio-ready scene folder containing extracted images, COLMAP outputs, and
`transforms.json`, which are then reused by both 3DGS training and SAM 2 mask
extraction.

For Gaussian Grouping style training, the repo now also includes:

- scene-global identity label remapping from SAM/DEVA-style mask manifests,
- a learnable Gaussian identity field with a `1x1` classifier head,
- a 3D spatial consistency regularizer for nearby Gaussian embeddings,
- renderer-output adapters and a single-step loss wrapper for future NerfStudio integration.

These pieces live under `gsegmenter/training/` and are intended to be wired into
the next identity-aware NerfStudio training step.
The repository now also includes an `IdentitySplatfactoModel` scaffold that can
emit `render_object` feature maps and consume `identity_labels` batches once the
NerfStudio data path is fully connected.
It now also includes an `IdentityFullImageDatamanager` plus
`build_identity_splatfacto_trainer_config(...)` so we can inject scene-global
identity labels without patching NerfStudio site-packages.
For local execution, use `scripts/train_identity_splatfacto.py` with the
NerfStudio-capable Python environment.
To inspect the learned identity field, use
`tools/render_identity_preview.py --load-config <config.yml> --frame-index <i> --output-path <png>`.
To export a `.ply` from the custom identity-aware checkpoint, use
`scripts/export_identity_splat.py --load-config <config.yml> --output-dir <dir>`.
For InteriorGS scene folders downloaded from Hugging Face, place a scene such as
`0001_839920/` under a local dataset root and validate it with
`scripts/inspect_interiorgs_scene.py --scene-root <scene-folder>`.
The compressed Gaussian file can then be expanded into a standard PLY with
`scripts/convert_interiorgs_ply.py --input-path <scene>/3dgs_compressed.ply --output-path <scene>/3dgs_uncompressed.ply`,
using PlayCanvas `splat-transform`.
Once converted, `scripts/build_interiorgs_groups.py --scene-root <scene> --output-root <dir>`
maps the dataset's `labels.json` boxes onto Gaussian centers so we can validate
grouping and editor behavior against InteriorGS ground-truth annotations before
reintroducing SAM-based lifting.
When predicted Gaussian object ids are available from SAM-based lifting,
`scripts/evaluate_gaussian_groups.py --gt-object-ids <gt.npy> --pred-object-ids <pred.npy>`
reports one-to-one IoU matches, unmatched group counts, and per-label recall
against the InteriorGS ground-truth grouping.

---

## 📌 Example Workflow

```bash
# 1. COLMAP pose estimation
python scripts/run_colmap.py --input data/scene01

# 2. Train 3D Gaussian Splatting with NerfStudio Splatfacto
python scripts/train_splatfacto.py --data-root data --scene-name scene01

# 3. Extract SAM 2 masks from a separate SAM 2 environment
python scripts/run_sam2_masks.py \
  --python-bin C:\envs\sam2\Scripts\python.exe \
  --images-dir data/scene01/images \
  --output-root outputs/scene01/masks \
  --checkpoint-path checkpoints/sam2.pt \
  --model-config sam2_hiera_l.yaml

# 4. Build Gaussian groups from multi-view masks
python scripts/lift_masks_to_gaussians.py \
  --dataset-root data/scene01 \
  --ply-path exports/scene01/splat.ply \
  --masks-root outputs/scene01/masks \
  --output-root outputs/scene01/lifting

# 5. Launch interactive editor
python editor/app.py \
  --scene outputs/scene01/checkpoint.pth \
  --groups outputs/scene01/groups
```

For a real room capture recorded as `.mp4`, the same flow starts with:

```bash
# 0. Convert a handheld room video into a NerfStudio dataset
python scripts/prepare_video_scene.py \
  --video-path captures/my_room.mp4 \
  --data-root data \
  --scene-name my_room \
  --num-frames-target 300 \
  --matching-method sequential \
  --sfm-tool colmap

# 1. Train 3DGS on the prepared scene
python scripts/train_splatfacto.py \
  --data-path data/my_room \
  --scene-name my_room
```

---

## 🔬 Research Motivation

이 프로젝트는 “빠르게 렌더링되는 3D scene representation”과 “열린 환경에서의 객체 수준 분할” 사이의 간극을 줄이기 위한 시도입니다.

Gaussian Grouping은 3DGS의 실시간성과 SAM의 범용 분할 능력을 연결하여, **재구성, 분할, 편집을 하나의 표현 안에서 통합**하려는 방향을 제시합니다.

---

## 🗺️ Roadmap

- [ ] Multi-view mask consistency 개선
- [ ] Gaussian identity encoding 학습 모듈 고도화
- [ ] Interactive editor에서 gizmo 기반 UX 강화
- [ ] Object removal 이후 infilling 품질 향상
- [ ] Open-vocabulary text-guided grouping 실험
- [ ] Large-scale indoor / outdoor scene 지원 확대

---

## 📚 References

- Segment Anything Model (SAM)
- 3D Gaussian Splatting (3DGS)
- Gaussian Grouping: Segment and Edit Anything in 3D Scenes
- COLMAP
- NerfStudio
