# PLAN_GG

## 목적

이 문서는 현재 확보한 3DGS 학습 결과를 시작점으로 삼아, `Gaussian Grouping` 기반의 3D 객체 분리 파이프라인을 구현하기 위한 실행 계획을 정리한다.

최우선 목표는 다음과 같다.

1. 2D 마스크를 여러 시점에서 안정적으로 추출한다.
2. 각 마스크를 3D Gaussian 공간으로 lifting하여 Gaussian별 object identity를 부여한다.
3. 동일 객체가 여러 뷰에서 일관된 3D group으로 묶이도록 multiview aggregation을 설계한다.
4. 이후 interactive editor가 바로 사용할 수 있는 group metadata를 출력한다.

---

## 현재 확보된 입력 자산

현재 `library` 샘플에 대해 다음 입력이 준비된 상태다.

- NerfStudio dataset root: `C:\GSegmenter\data\nerfstudio\library`
- 학습 이미지: `C:\GSegmenter\data\nerfstudio\library\images`
- 카메라 및 프레임 정보: `C:\GSegmenter\data\nerfstudio\library\transforms.json`
- 학습 config: `C:\GSegmenter\outputs\library\splatfacto\2026-04-14_194808\config.yml`
- 체크포인트: `C:\GSegmenter\outputs\library\splatfacto\2026-04-14_194808\nerfstudio_models\step-000029999.ckpt`
- export된 Gaussian point set: `C:\GSegmenter\exports\library_splat\splat.ply`

이 계획에서는 우선 `export된 splat.ply + transforms.json + 원본 이미지` 조합을 기준 입력으로 사용한다.

---

## 선택한 방법

보수안 기준으로 아래 조합을 채택한다.

- 2D 분할: `SAM 2`
- 프레임 간 안정화: 기본은 생략, 필요 시 `DEVA`를 선택적으로 추가
- 3D grouping: `Gaussian Grouping` 스타일의 multiview identity lifting
- 텍스트 라벨링: 1차 구현에서는 제외, grouping 완료 후 `OpenCLIP`을 post-hoc으로 추가

이 선택의 이유는 현재 단계에서 가장 중요한 것이 `최신 모델 채택`이 아니라 `2D mask -> 3D identity -> editor input` 경로를 안정적으로 끝까지 완성하는 것이기 때문이다.

---

## 전체 파이프라인

### Stage 0. 입력 정규화

목표는 이후 모듈들이 동일한 scene 표현을 공유하도록 최소 공통 scene descriptor를 만드는 것이다.

입력:

- `transforms.json`
- `splat.ply`
- `images/*.png`

출력:

- `scene_manifest.json`
- frame index와 image path, camera intrinsics/extrinsics를 정리한 메타데이터
- Gaussian 배열의 기본 속성 테이블

구현 방향:

- `gsegmenter/data/nerfstudio_scene.py`
- `gsegmenter/mapping/gaussian_io.py`
- `scripts/prepare_scene_manifest.py`

핵심 체크:

- NerfStudio camera 좌표계와 우리가 사용할 world 좌표계를 명시적으로 문서화한다.
- 각 frame의 `file_path`, `intrinsics`, `camera_to_world` shape를 고정한다.
- Gaussian 속성은 최소한 `xyz`, `scale`, `rotation`, `opacity`, `color/sh`를 읽을 수 있어야 한다.

---

### Stage 1. SAM 2 기반 2D 마스크 추출

목표는 학습에 사용된 모든 뷰에 대해 instance mask 후보를 생성하는 것이다.

입력:

- `images/*.png`

출력:

- `outputs/library/masks/frame_XXXXX/*.png`
- `outputs/library/masks/frame_XXXXX/instances.json`

권장 저장 포맷:

- mask binary PNG
- 각 instance별 `instance_id`, `bbox_xyxy`, `score`, `area`, `mask_path`

구현 방향:

- `gsegmenter/segmentation/sam2_extractor.py`
- `scripts/extract_sam2_masks.py`

설계 원칙:

- 1차 구현은 자동 분할 기반으로 간다.
- frame별 결과를 독립 저장해서 재실행 비용을 줄인다.
- 추후 DEVA를 붙일 수 있도록 `frame_id -> instance list` 구조를 고정한다.

검증 기준:

- 임의의 10장 샘플에서 큰 foreground object가 안정적으로 분할되는지 시각 점검
- mask 면적이 지나치게 작거나 화면 전체를 덮는 instance 비율을 통계로 기록

---

### Stage 2. 2D mask quality filtering

SAM 출력은 그대로 lifting하면 노이즈가 많을 수 있으므로, 먼저 2D 단계에서 후보를 정리한다.

입력:

- frame별 instance masks

출력:

- 필터링된 mask manifest
- 폐기된 mask 로그

필터 규칙 초안:

- 최소 면적 threshold
- 최대 면적 threshold
- 너무 가는 strip 형태 제거
- confidence threshold
- 이미지 경계에 과도하게 붙은 mask 표시

구현 방향:

- `gsegmenter/segmentation/filtering.py`
- `scripts/filter_masks.py`

검증 기준:

- 제거율 통계
- 유지된 mask들의 평균 면적, instance 수, frame별 편차

---

### Stage 3. Gaussian visibility / projection 준비

목표는 각 Gaussian이 각 카메라에서 어디에 투영되는지, 그리고 실제로 보이는지 계산하는 것이다.

입력:

- `splat.ply`
- `transforms.json`

출력:

- frame별 visible Gaussian index
- frame별 projected 2D coordinate
- depth ordering 또는 visibility score

구현 방향:

- `gsegmenter/render/projection.py`
- `gsegmenter/mapping/visibility.py`

핵심 계산:

- Gaussian center를 camera frame으로 변환
- image plane 투영
- 이미지 범위 내부 여부 확인
- near/far 및 depth 유효성 체크

주의:

- 1차 구현은 Gaussian center 기반 근사 투영으로 시작한다.
- 이후 필요하면 anisotropic footprint나 rasterized contribution 기반으로 고도화한다.
- 좌표계와 matrix convention은 함수 docstring에 명시한다.

검증 기준:

- 랜덤 frame에서 projected center가 실제 이미지 위치와 대략 일치하는지 overlay 확인
- invalid projection 비율과 behind-camera 비율 기록

---

### Stage 4. 2D mask -> 3D Gaussian vote lifting

목표는 각 mask가 어떤 Gaussian들을 지지하는지 vote를 누적하는 것이다.

입력:

- filtered mask set
- frame별 projected Gaussian coordinates

출력:

- Gaussian별 vote histogram
- `(gaussian_index, frame_id, instance_id, vote_weight)` 로그

기본 아이디어:

- 어떤 Gaussian이 특정 frame에서 어떤 mask 내부에 투영되면 그 mask의 local instance label에 vote를 부여한다.
- weight는 단순 binary부터 시작하고, 이후 다음 요소를 반영할 수 있다.
  - mask confidence
  - Gaussian opacity
  - depth 안정성
  - reprojection confidence

구현 방향:

- `gsegmenter/mapping/lifting.py`
- `scripts/lift_masks_to_gaussians.py`

핵심 난점:

- 2D instance ID는 frame마다 지역적이므로 곧바로 global object ID가 아니다.
- 따라서 여기 단계의 출력은 `global object label`이 아니라 `multiview association을 위한 vote evidence`여야 한다.

검증 기준:

- 임의 Gaussian 샘플에 대해 여러 frame에서 어떤 mask들에 들어갔는지 추적 가능해야 한다.
- foreground object에 속한 Gaussian들은 반복적으로 비슷한 mask 계열에 매핑되어야 한다.

---

### Stage 5. Multiview instance association

목표는 frame-local mask instance들을 scene-level object hypothesis로 통합하는 것이다.

입력:

- lifting vote evidence
- frame별 mask geometry
- 필요 시 mask appearance feature

출력:

- `global_object_id`
- frame-local instance와 global object의 매핑 테이블

추천 방식:

- 1차는 graph-based association
- node: frame-local instance
- edge score:
  - 공유 Gaussian 비율
  - Gaussian vote overlap
  - 뷰 간 appearance similarity
  - 카메라 인접성

초기 알고리즘:

1. frame-local instance를 node로 만든다.
2. 인접 frame 또는 overlap이 큰 view pair 사이에서 edge를 만든다.
3. shared Gaussian overlap이 큰 node끼리 union-find 또는 connected component로 묶는다.
4. 지나치게 큰 component는 후처리로 다시 분할한다.

구현 방향:

- `gsegmenter/mapping/association.py`
- `scripts/associate_multiview_instances.py`

검증 기준:

- 동일한 물체가 여러 frame에서 하나의 global object로 묶이는지 수작업 샘플 검토
- object purity와 fragmentation 수치 기록

---

### Stage 6. Gaussian Group 생성

목표는 Gaussian별 최종 object ID를 확정하고 editor가 쓰기 쉬운 group representation으로 변환하는 것이다.

입력:

- global object hypotheses
- Gaussian vote histogram

출력:

- `outputs/library/groups/gaussian_groups.json`
- `outputs/library/groups/gaussian_object_ids.npy`
- object별 Gaussian index list
- object별 bbox / centroid / dominant frames / support count

할당 규칙 초안:

- Gaussian마다 가장 강한 global object hypothesis를 primary label로 배정
- 낮은 confidence Gaussian은 `unknown` 또는 `background`로 남김
- 작은 isolated cluster는 post-process로 제거하거나 인접 group에 병합

구현 방향:

- `gsegmenter/mapping/grouping.py`
- `scripts/build_gaussian_groups.py`

검증 기준:

- object별 Gaussian 개수 분포
- 지나치게 조각난 group 수
- 시각화했을 때 객체 경계가 직관적으로 분리되는지 확인

---

### Stage 7. 시각화 및 디버깅 도구

Grouping 파이프라인은 중간 단계가 많기 때문에, 최소한의 디버깅 도구가 필수다.

필수 시각화:

- 2D image 위에 projected Gaussian center overlay
- 2D mask와 Gaussian vote heatmap overlay
- object별 colorized Gaussian point cloud
- frame별 selected object reproject visualization

구현 방향:

- `tools/visualize_masks.py`
- `tools/visualize_lifting.py`
- `tools/visualize_groups.py`

---

## 산출물 포맷 제안

최종적으로 editor와 downstream task가 바로 사용할 수 있도록 아래 산출물을 고정한다.

### 1. Scene manifest

- scene name
- source dataset root
- image count
- frame metadata
- coordinate convention 설명

### 2. Mask manifest

- frame id
- per-instance metadata
- mask path
- quality stats

### 3. Group manifest

- global object id
- Gaussian count
- centroid
- bbox
- support frames
- confidence
- optional text label slot

### 4. Gaussian object assignment

- `N`개 Gaussian에 대한 object id 배열
- unknown/background 분리 규칙 포함

---

## 구현 우선순위

### Phase A. 최소 동작 경로

먼저 아래 경로를 가장 짧게 완성한다.

1. `splat.ply` 읽기
2. `transforms.json` 읽기
3. Gaussian center를 frame으로 투영
4. frame별 SAM 2 mask 추출
5. mask 내부 hit 기반 Gaussian vote 누적
6. 가장 단순한 overlap 기반 multiview association
7. Gaussian group export

이 단계의 목적은 `정확도 최고화`가 아니라 `end-to-end working pipeline` 확보다.

### Phase B. 정확도 개선

- visibility 계산 개선
- depth-aware weighting
- noisy mask filtering 개선
- 작은 fragment 제거
- object boundary refinement

### Phase C. 확장 기능

- DEVA 기반 temporal consistency
- OpenCLIP 기반 object naming
- editor selection API 연결

---

## 검증 계획

정량 지표와 정성 검토를 함께 사용한다.

### 정량

- frame당 평균 instance 수
- Gaussian vote coverage
- labeled Gaussian 비율
- object fragmentation count
- object purity 추정치

### 정성

- colorized group point cloud 시각화
- 특정 객체를 여러 frame에 재투영했을 때 일관성 확인
- furniture, wall, floor 같은 큰 구조물이 과도하게 섞이지 않는지 확인

---

## 위험 요소와 대응

### 1. 2D 마스크 과분할 / 미분할

대응:

- Stage 2 filtering 강화
- 필요 시 prompt 기반 재추론 또는 DEVA 추가

### 2. 투영 기반 오할당

대응:

- center-only projection으로 시작하되 depth와 opacity weighting 추가
- 이후 rasterized contribution 기반 visibility로 확장

### 3. 같은 객체가 여러 group으로 쪼개짐

대응:

- multiview association 단계에서 shared Gaussian overlap을 핵심 edge feature로 사용
- post-merge heuristic 추가

### 4. 배경 구조물과 foreground 혼합

대응:

- low-confidence Gaussian을 강제 할당하지 않음
- background / unknown label을 명시적으로 유지

---

## 바로 다음 구현 작업

다음 작업은 아래 순서로 진행하는 것이 가장 효율적이다.

1. `gsegmenter/data/nerfstudio_scene.py`
   `transforms.json`과 image 목록을 읽어 공통 scene descriptor를 만든다.

2. `gsegmenter/mapping/gaussian_io.py`
   `splat.ply`에서 Gaussian 중심과 기본 속성을 읽는다.

3. `gsegmenter/render/projection.py`
   Gaussian center를 각 frame으로 투영하는 최소 함수를 만든다.

4. `scripts/prepare_scene_manifest.py`
   위 세 모듈을 연결해 재사용 가능한 scene manifest를 출력한다.

5. `gsegmenter/segmentation/sam2_extractor.py`
   image 폴더 전체에 대해 SAM 2 inference를 수행한다.

이 5개가 준비되면 Gaussian Grouping 구현을 위한 핵심 기반이 완성된다.
