## 1. 폴더 트리 구조 (Folder Tree Structure)

연구용 스크립트와 프로덕션 코드가 섞이지 않도록 모듈의 목적과 성격에 따라 디렉터리를 분리합니다.

```text
GSegmenter/
├── configs/                     # YAML 및 Hydra 기반 설정 파일 모음
│   ├── base.yaml
│   ├── dataset/
│   ├── training/
│   ├── mapping/
│   └── editor/
├── gsegmenter/                  # 메인 패키지
│   ├── data/                    # [Infrastructure] Data Acquisition
│   ├── training/                # [Infrastructure] Gaussian Training
│   ├── segmentation/            # [Research] Mask Extraction (SAM 등)
│   ├── mapping/                 # [Research] Gaussian Grouping & Lifting
│   ├── editor/                  # [Stable] Editor Logic & Geometry Infilling
│   ├── render/                  # 렌더링 래퍼 및 프로젝션 유틸리티
│   ├── utils/                   # 로깅, 수학, 좌표계 변환 등 공통 유틸
│   └── conf.py                  # Dataclass 기반의 Config 정의
├── scripts/                     # 실행 스크립트 (학습, 전처리, UI 실행 등)
├── notebooks/                   # [Research] 결과 분석, 시각화, 실험용 주피터 노트북
├── tests/                       # 단위 테스트 (Projection, Transform 등)
├── README.md
├── AGENTS.md
└── requirements.txt
```

---

## 2. 각 모듈의 책임 및 상세 설계

각 모듈은 명확한 입력과 출력을 가지며, 개별 모듈의 테스트 가능성을 확보합니다. 인프라성 모듈(안정적)과 연구성 모듈(실험적)을 나누어 관리합니다.

### 2.1. `data/` (Data Acquisition) - **[안정적인 인프라 모듈]**
* **책임**: COLMAP 등 데이터셋 출력 결과를 로딩, 파싱, 전처리합니다. 명시적인 3D 좌표계를 관리하며 다른 모듈에서 포맷에 종속되지 않는 일관된 Scene 데이터를 제공합니다.
* **주요 입력**: COLMAP sparse 파일(`cameras.bin`, `images.bin`, `points3D.bin`), 원본 이미지.
* **주요 출력**: 파싱된 카메라 Intrinsic/Extrinsic, 초기 Sparse Point Cloud, 프레임 매니저(Data Loader).
* **예상 의존성**: `numpy`, `opencv-python`, `colmap` (파싱 라이브러리), `torch`.

### 2.2. `training/` (Gaussian Training) - **[안정적인 인프라 모듈]**
* **책임**: NerfStudio 스타일과의 호환성을 고려한 3D Gaussian Splatting 핵심 훈련 루프. 역전파를 통한 최소화 등 파라미터(Mean, Covariance, Opacity, SH) 최적화에만 집중합니다.
* **주요 입력**: `data/`로부터 받은 Scene 데이터 텐서, 학습 하이퍼파라미터.
* **주요 출력**: 3DGS 체크포인트(`ply` 또는 `pt` 파일 포맷), 텐서보드/WandB 로그.
* **예상 의존성**: `torch`, `diff-gaussian-rasterization` (혹은 gsplat).

### 2.3. `segmentation/` (Mask Extraction) - **[연구 성격 모듈]**
* **책임**: 다방향(Multi-view) 2D 이미지에 대한 SAM(Segment Anything Model) 추론. 향후 다양한 Video/2D 세그멘테이션 모델로 대체될 수 있도록 인터페이스를 분리합니다.
* **주요 입력**: RGB 이미지, 선택적 프롬프트(Point, Bounding Box 등).
* **주요 출력**: 2D 세그멘테이션 마스크(Binary or Probability), 마스크 피처 임베딩.
* **예상 의존성**: `segment-anything` (혹은 관련 비전 모델), `torchvision`.

### 2.4. `mapping/` (Gaussian Grouping & Identity) - **[연구 성격 모듈 (가장 핵심/변동성 높음)]**
* **책임**: 생성된 2D 마스크를 3D 공간의 Gaussian Entity들에 매핑(Lifting)하고, 다중 뷰에서 식별된 마스크 간의 일관성을 맞추기 위해 결합(Aggregation) 및 투표(Voting) 로직을 수행합니다. 
* **주요 입력**: 학습이 완료된 3D Gaussians 데이터 구조, 마스크된 2D 이미지 세트, 해당 프레임의 카메라 Poses.
* **주요 출력**: 각 Gaussian point 단위의 Object ID (또는 Group Label), 3D 상에서의 오브젝트 경계 등 메타데이터.
* **예상 의존성**: `torch`, 클러스터링 기반 라이브러리(`scikit-learn`의 HDBSCAN 등), 자체 작성한 고속 CUDA 프로젝션 커널.

### 2.5. `editor/` (Editor Logic & Infilling) - **[안정적인 프로덕션 및 UX 모듈]**
* **책임**: 실시간 오브젝트 조작(이동, 회전, 크기 조절, 삭제) 및 상태를 관리합니다. Local World 변환과 Gaussian 내부 파라미터 변환을 명확히 구분하여 되돌리기(Reversible)가 가능해야 합니다.
* **주요 입력**: Object ID가 부여된 3D Gaussians 데이터 상태 트리, 유저 인터랙션 커맨드(Transform Matrices, Select/Delete 이벤트).
* **주요 출력**: 업데이트된 Gaussian 상태(위치, 공분산 재계산 등 결괏값), 조작 이후 렌더링된 실시간 프레임, 기하학적 Infilling이 적용된 데이터.
* **예상 의존성**: `viser` 등 웹 기반 실시간 UI 통신 프레임워크, 공간 변환 행렬 관련 라이브러리(`transforms3d` 혹은 `pytorch3d`).

---

## 3. Config 구조 및 네이밍 규칙

**Config 구조 방향성**
* YAML 기반의 계층적 구조를 가져가되, 코드 상에서는 Type-Hinting이 지원되는 Python `dataclass`를 활용하여 구성합니다 (예: Hydra 혹은 OmegaConf 사용).
* 하드코딩(로컬 경로, Magic Number)은 철저히 배제합니다.

**네이밍 및 코드 규약**
* **모듈/함수/변수**: 명확한 그래픽스 및 비전 Semantics를 담은 Snake Case.
  * *좋은 예시*: `project_mask_to_gaussians`, `aggregate_multiview_votes`, `apply_object_transform`
  * *나쁜 예시*: `process_data`, `update_info`
* **클래스 및 데이터 모델**: 파스칼 케이스(Pascal Case).
  * *예시*: `GaussianModel`, `MaskFeatureExtractor`
* **명시적 접미어(Suffix) 규칙**: 다차원 배열 및 복잡한 State를 명확히 하기 위한 약어.
  * Torch 텐서 배열: `_tensor` 혹은 `_mat` (예: `rotation_mat_tensor`)
  * 환경 설정 값: `_cfg`

---

## 4. 상위 모듈 분리의 설계 이유 (Design Rationale)

1. **학습(Training)과 매핑(Mapping)의 분리**
   * 일반적인 3DGS는 정적이고 빠른 렌더링에 집중하지만, 우리의 목적은 '객체 분리와 식별'에 있습니다. `training`은 순수 광도(photometric) 최적화에만 집중하고, `mapping`을 별도로 두어 2D 프라이어 모델이 변경되어도 기존 파이프라인이 망가지지 않는 느슨한 결합(Loose Coupling)을 지향했습니다.

2. **UX 상태 관리(Editor)와 순수 기하학(Render)의 분리**
   * 에디터는 유저가 언제든 취소/복구 및 조작할 수 있는 **상태(State)** 중심적인 성격을 띱니다. `editor` 로직이 커널 수준의 렌더링 폴더로 스며들면 시스템 복잡도가 기하급수적으로 높아집니다.
   * `editor`는 오직 '어느 트랜스폼 매트릭스를 적용할 것인지'를 관장하고, 적용 계산은 유틸리티나 렌더 모듈에서 수행함으로써 무거운 연산을 유저 이벤트 훅단에서 분리할 수 있습니다.

3. **명시적인 연구(Research) 공간 격리**
   * Multi-view Semantic Lifting (2D -> 3D 할당 알고리즘 등)은 매우 실험적인 성격이 강합니다. 시스템 안정성을 해치기 쉽기 때문에 인프라 성격의 코드(데이터 로드 및 기본 학습)가 있는 폴더와 `mapping` / `segmentation` 폴더를 구분하여 **실무진(인프라 최적화)과 연구진(알고리즘 향상)의 충돌 지검 없는 병렬적 개선**이 가능하도록 설계했습니다.