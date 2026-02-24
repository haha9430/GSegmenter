# Proposed Project Structure for GSegmenter

README.md에서 설명한 파이프라인(데이터 수집 → 3DGS 학습 → 2D/3D ID 매핑 → 인터랙티브 편집기)을 기준으로, 아래와 같은 구조를 추천합니다.

```text
GSegmenter/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data/
│   │   ├── colmap.yaml
│   │   └── dataset_paths.yaml
│   ├── training/
│   │   ├── gaussian_splatting.yaml
│   │   └── nerfstudio.yaml
│   ├── mapping/
│   │   ├── sam_inference.yaml
│   │   └── voting.yaml
│   └── editor/
│       ├── transform.yaml
│       └── infilling.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── colmap/
│   ├── masks_2d/
│   └── scene_3d/
├── models/
│   ├── checkpoints/
│   └── pretrained/
├── src/
│   └── gsegmenter/
│       ├── __init__.py
│       ├── pipelines/
│       │   ├── data_acquisition.py
│       │   ├── train_3dgs.py
│       │   ├── identity_mapping.py
│       │   └── interactive_editing.py
│       ├── segmentation/
│       │   ├── sam_wrapper.py
│       │   └── mask_postprocess.py
│       ├── gaussian/
│       │   ├── gaussian_io.py
│       │   ├── transform_ops.py
│       │   └── infilling.py
│       ├── rendering/
│       │   ├── realtime_renderer.py
│       │   └── viewer_bridge.py
│       ├── editor/
│       │   ├── object_selector.py
│       │   ├── gizmo_controller.py
│       │   └── edit_history.py
│       ├── utils/
│       │   ├── io.py
│       │   ├── math3d.py
│       │   └── logging.py
│       └── cli/
│           ├── preprocess.py
│           ├── train.py
│           ├── map_id.py
│           └── edit.py
├── scripts/
│   ├── run_colmap.sh
│   ├── train_3dgs.sh
│   ├── run_mapping.sh
│   └── launch_editor.sh
├── tests/
│   ├── test_transform_ops.py
│   ├── test_identity_mapping.py
│   └── test_infilling.py
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   └── 02_mapping_debug.ipynb
├── outputs/
│   ├── renders/
│   ├── logs/
│   └── experiments/
└── docs/
    ├── architecture.md
    ├── data_format.md
    └── editor_workflow.md
```

## 디렉터리 설계 의도

- `configs/`: 실험 재현성과 환경 분리를 위해 단계별 설정을 분리합니다.
- `data/`: 원본/중간 산출물/최종 산출물을 분리하여 파이프라인 디버깅을 쉽게 합니다.
- `src/gsegmenter/pipelines/`: README의 시스템 아키텍처 4단계를 코드 구조와 1:1로 맞춥니다.
- `src/gsegmenter/segmentation`, `gaussian`, `rendering`, `editor`: 핵심 도메인 기능을 모듈 경계로 분리해 유지보수성을 높입니다.
- `scripts/`: 반복 실행되는 end-to-end 작업을 단일 명령으로 운영하기 좋습니다.
- `tests/`: 변환 행렬, ID 매핑, 경계 infilling 등 실패 위험이 높은 핵심 로직을 우선 검증합니다.
- `docs/`: 아키텍처와 입출력 포맷 문서를 분리해 협업 온보딩 속도를 높입니다.

## 파일/디렉터리별 상세 역할

### 루트 파일
- `README.md`
  - 프로젝트 개요, 핵심 기능, 아키텍처, 설치 방법을 제공하는 엔트리 문서입니다.
  - 신규 기여자에게 “이 프로젝트가 무엇을 하는지”를 가장 빠르게 전달합니다.
- `requirements.txt`
  - 런타임 의존성의 최소 집합을 고정합니다.
  - 실험 환경 재현 시 `pip install -r requirements.txt` 기준점이 됩니다.
- `pyproject.toml`
  - 패키지 메타데이터, 빌드 시스템, 린트/포맷터/테스트 도구 설정을 중앙집중 관리합니다.

### `configs/`
- `configs/data/colmap.yaml`
  - COLMAP 실행 파라미터(특징 추출 옵션, 매칭 전략, BA 설정 등)를 정의합니다.
- `configs/data/dataset_paths.yaml`
  - 입력 이미지, 캘리브레이션, 결과 출력 경로를 환경별로 분리 관리합니다.
- `configs/training/gaussian_splatting.yaml`
  - 3DGS 학습 하이퍼파라미터(learning rate, densification, pruning 정책)를 저장합니다.
- `configs/training/nerfstudio.yaml`
  - NerfStudio 실행 옵션(데이터 파서, trainer, viewer)을 통합 관리합니다.
- `configs/mapping/sam_inference.yaml`
  - SAM 추론 설정(모델 타입, 프롬프트 방식, threshold)을 관리합니다.
- `configs/mapping/voting.yaml`
  - 멀티뷰 마스크를 3D ID로 합의(voting)하는 규칙을 정의합니다.
- `configs/editor/transform.yaml`
  - 객체 이동/회전/스케일 제약(축 잠금, 스냅, 좌표계 기준)을 설정합니다.
- `configs/editor/infilling.yaml`
  - 객체 제거/이동 후 경계 복원(infilling) 알고리즘의 반경·밀도 기준을 정의합니다.

### `data/`
- `data/raw/`
  - 원본 촬영 이미지/영상 프레임을 보존하는 불변 데이터 저장소입니다.
- `data/processed/`
  - 전처리(리사이즈, undistortion, frame sampling) 완료된 학습 입력을 저장합니다.
- `data/colmap/`
  - COLMAP 산출물(카메라 파라미터, sparse/dense 재구성)을 저장합니다.
- `data/masks_2d/`
  - SAM 또는 수동 보정으로 생성한 2D 마스크를 저장합니다.
- `data/scene_3d/`
  - 3DGS용 장면 표현(포인트/가우시안 속성, ID 부착 결과)을 저장합니다.

### `models/`
- `models/checkpoints/`
  - 학습 중간/최종 체크포인트를 버전별로 보관합니다.
- `models/pretrained/`
  - 외부 사전학습 모델(SAM backbone 등)을 관리합니다.

### `src/gsegmenter/`
- `__init__.py`
  - 패키지 버전/공용 API 노출 지점을 정의합니다.

#### `src/gsegmenter/pipelines/`
- `data_acquisition.py`
  - 입력 데이터 정합성 검사, COLMAP 호출, 결과 인덱싱까지를 오케스트레이션합니다.
- `train_3dgs.py`
  - 3D Gaussian Splatting 학습 루프를 실행하고 체크포인트를 저장합니다.
- `identity_mapping.py`
  - 2D 멀티뷰 마스크를 3D 가우시안 ID로 투영·합의하는 핵심 파이프라인입니다.
- `interactive_editing.py`
  - 선택/변환/삭제/복원(infilling) 과정을 세션 단위로 연결합니다.

#### `src/gsegmenter/segmentation/`
- `sam_wrapper.py`
  - SAM 모델 로딩, 추론, 배치 처리 인터페이스를 표준화합니다.
- `mask_postprocess.py`
  - 작은 조각 제거, hole filling, 클래스 병합 등 마스크 후처리를 담당합니다.

#### `src/gsegmenter/gaussian/`
- `gaussian_io.py`
  - 가우시안 속성(position, covariance, color, opacity) 직렬화/역직렬화를 담당합니다.
- `transform_ops.py`
  - 객체 단위 rigid transform 및 공분산 갱신 수학 연산을 구현합니다.
- `infilling.py`
  - 편집 후 생긴 공간적 결손을 주변 기하 기반으로 보정합니다.

#### `src/gsegmenter/rendering/`
- `realtime_renderer.py`
  - 편집 상태를 반영해 실시간 렌더링 프레임을 생성합니다.
- `viewer_bridge.py`
  - 렌더러와 UI 뷰어(웹/데스크톱) 사이 이벤트·상태 전달을 담당합니다.

#### `src/gsegmenter/editor/`
- `object_selector.py`
  - 클릭/드래그/ID 기반 선택 로직과 멀티 선택 정책을 관리합니다.
- `gizmo_controller.py`
  - UI gizmo 입력을 변환 행렬로 변환해 대상 객체에 적용합니다.
- `edit_history.py`
  - undo/redo, 세션 스냅샷, 편집 명령 로그를 관리합니다.

#### `src/gsegmenter/utils/`
- `io.py`
  - 파일 경로, 캐시, 공통 직렬화(JSON/YAML/NPY) 유틸을 제공합니다.
- `math3d.py`
  - 좌표계 변환, 쿼터니언/행렬 변환, 투영/역투영 공통 함수를 제공합니다.
- `logging.py`
  - 단계별 구조화 로그/실험 태그 기록을 일관되게 관리합니다.

#### `src/gsegmenter/cli/`
- `preprocess.py`
  - 원본 데이터 전처리와 COLMAP 준비를 위한 CLI 진입점입니다.
- `train.py`
  - 3DGS 학습 실행용 CLI입니다.
- `map_id.py`
  - 2D→3D ID 매핑 실행용 CLI입니다.
- `edit.py`
  - 인터랙티브 편집기 실행/세션 로드 CLI입니다.

### `scripts/`
- `run_colmap.sh`
  - 재구성 파이프라인을 반복 가능하게 실행하는 배치 스크립트입니다.
- `train_3dgs.sh`
  - 대표 학습 설정으로 3DGS 훈련을 시작하는 실행 스크립트입니다.
- `run_mapping.sh`
  - SAM 추론부터 voting 기반 ID 매핑까지 일괄 실행합니다.
- `launch_editor.sh`
  - 렌더러/뷰어/백엔드 의존 서비스를 한 번에 기동합니다.

### `tests/`
- `test_transform_ops.py`
  - 변환 행렬 적용 전후 위치/공분산 보존 성질을 검증합니다.
- `test_identity_mapping.py`
  - 멀티뷰 충돌 상황에서 voting 결과의 안정성을 검증합니다.
- `test_infilling.py`
  - 결손 영역 복원 품질(연결성, 밀도 연속성)을 검증합니다.

### `notebooks/`
- `01_data_inspection.ipynb`
  - 입력 데이터 품질 점검(블러/노출/시점 분포)과 샘플 시각화를 수행합니다.
- `02_mapping_debug.ipynb`
  - 2D 마스크가 3D에 어떻게 올라가는지 디버깅용 시각화를 제공합니다.

### `outputs/`
- `outputs/renders/`
  - 정적 프레임/비교 이미지 등 렌더 결과물을 저장합니다.
- `outputs/logs/`
  - 학습/추론/편집 실행 로그를 수집합니다.
- `outputs/experiments/`
  - 실험별 메트릭, 설정 스냅샷, 아티팩트를 버전 단위로 보관합니다.

### `docs/`
- `docs/architecture.md`
  - 시스템 컴포넌트 관계와 데이터 흐름을 다이어그램 중심으로 설명합니다.
- `docs/data_format.md`
  - 입력/중간/출력 포맷(좌표계, 단위, 파일 스키마)을 명세합니다.
- `docs/editor_workflow.md`
  - 사용자 편집 시나리오(선택→변형→검증→저장)를 단계별로 문서화합니다.

## 시작 순서 권장

1. `src/gsegmenter/pipelines/`와 `cli/`부터 뼈대를 생성
2. `segmentation`/`gaussian`의 최소 기능 구현
3. `tests/`에 핵심 수학/매핑 로직 테스트 우선 작성
4. `scripts/`로 파이프라인 자동화
5. `docs/`에 데이터 포맷/편집 워크플로우 고정
