# GSegmenter 구현 시작 계획 (Execution Plan)

`PROPOSED_STRUCTURE.md`에서 정의한 구조를 그대로 유지하면서, 실제 개발을 시작하기 위한 우선순위 계획입니다.

---

## 0) 목표와 원칙

### 목표
- 가장 빠르게 **동작하는 최소 파이프라인(MVP)** 을 만든다.
- 이후 단계적으로 품질(정확도, 속도, 안정성)을 높인다.

### 원칙
- **파이프라인 단위 통합**: 모듈별 완성보다 E2E(입력→결과) 연결을 우선.
- **재현성 우선**: 모든 실행은 config + CLI + logs로 남긴다.
- **테스트 선행**: 수학/매핑 핵심 로직은 구현과 동시에 테스트 작성.

---

## 1) 1주차: 프로젝트 뼈대 + 실행 진입점 확보

### 1-1. 스캐폴딩
- 생성 대상:
  - `src/gsegmenter/` 하위 패키지 (`pipelines`, `segmentation`, `gaussian`, `rendering`, `editor`, `utils`, `cli`)
  - `configs/` 하위 기본 yaml
  - `scripts/`, `tests/`, `docs/` 기본 파일
- 완료 기준:
  - Python import 에러 없이 패키지 로딩 가능
  - CLI 도움말 실행 가능 (`python -m gsegmenter.cli.train --help` 형태)

### 1-2. 공통 인프라
- `utils/io.py`, `utils/logging.py`, `utils/math3d.py` 최소 구현
- 공통 config loader(yaml), 실험 출력 디렉터리 생성 규칙 확정
- 완료 기준:
  - config 1개를 읽고 logs/outputs 경로를 자동 생성하는 샘플 실행 성공

---

## 2) 2주차: Data Acquisition + 3DGS 학습 최소 루프

### 2-1. 데이터 준비 파이프라인
- `pipelines/data_acquisition.py`, `cli/preprocess.py`, `scripts/run_colmap.sh`
- 역할:
  - 입력 데이터 검증
  - COLMAP 실행 래핑(또는 실행 결과 읽기)
  - 결과 메타데이터 정리
- 완료 기준:
  - 샘플 씬 1개에서 `data/colmap/` 산출물 경로 인덱싱 완료

### 2-2. 3DGS 학습 엔트리
- `pipelines/train_3dgs.py`, `cli/train.py`, `scripts/train_3dgs.sh`
- 역할:
  - 학습 config 로드
  - 체크포인트 저장
  - 기본 metric/log 출력
- 완료 기준:
  - 최소 스텝 학습이 종료되고 `models/checkpoints/`에 결과 저장

---

## 3) 3주차: 2D→3D Identity Mapping MVP

### 3-1. SAM 추론 + 후처리
- `segmentation/sam_wrapper.py`, `segmentation/mask_postprocess.py`
- 완료 기준:
  - 샘플 이미지 배치에 대해 마스크 생성 및 후처리 결과 저장

### 3-2. Voting 기반 매핑
- `pipelines/identity_mapping.py`, `cli/map_id.py`, `configs/mapping/voting.yaml`
- 완료 기준:
  - 멀티뷰 마스크가 3D 가우시안 ID에 매핑되어 `data/scene_3d/`에 저장

---

## 4) 4주차: Interactive Editing MVP

### 4-1. 선택/변환 기초
- `editor/object_selector.py`, `editor/gizmo_controller.py`, `gaussian/transform_ops.py`
- 완료 기준:
  - 객체 ID 선택 후 translate/rotate가 가우시안 상태에 반영

### 4-2. 편집 세션/이력
- `editor/edit_history.py`, `pipelines/interactive_editing.py`, `cli/edit.py`
- 완료 기준:
  - 최소 undo/redo 동작
  - 세션 저장/로드 가능

---

## 5) 5주차: Infilling + 렌더/뷰어 연결

### 5-1. Infilling
- `gaussian/infilling.py`, `configs/editor/infilling.yaml`
- 완료 기준:
  - 삭제/이동 후 결손 영역 복원 함수 실행 및 결과 비교 가능

### 5-2. 렌더링 브리지
- `rendering/realtime_renderer.py`, `rendering/viewer_bridge.py`, `scripts/launch_editor.sh`
- 완료 기준:
  - 편집 결과를 시각적으로 확인 가능한 루프 확보

---

## 6) 테스트 전략 (동시에 진행)

### 우선 테스트 파일
- `tests/test_transform_ops.py`
  - rigid transform 적용 후 위치/공분산 연산의 수학적 일관성 검증
- `tests/test_identity_mapping.py`
  - 멀티뷰 충돌 상황에서 voting 결과 안정성 검증
- `tests/test_infilling.py`
  - infilling 전후 연결성/밀도 연속성의 기본 검증

### 테스트 기준
- 단위 테스트: 수학·매핑 핵심 로직 커버
- 스모크 테스트: CLI 단위 최소 실행
- 회귀 테스트: 주요 버그 재현 입력 1~2개 고정

---

## 7) 마일스톤 정의

### Milestone A (MVP-E2E)
- preprocess → train → map_id → edit 파이프라인이 한 씬에서 끝까지 실행

### Milestone B (Usable Editor)
- 객체 선택/이동/회전/undo/redo + 기본 infilling 지원

### Milestone C (Stabilization)
- 성능 튜닝, 예외 처리 강화, 문서/테스트 보강

---

## 8) 지금 당장 시작할 To-do (우선순위 Top 10)

1. `pyproject.toml`/`requirements.txt` 초안 작성
2. `src/gsegmenter/__init__.py` 및 패키지 폴더 생성
3. `utils/io.py`, `utils/logging.py` 최소 구현
4. `configs/*` 기본 yaml 템플릿 생성
5. `cli/preprocess.py`, `cli/train.py`, `cli/map_id.py`, `cli/edit.py` argparse 뼈대 작성
6. `pipelines/data_acquisition.py` 스텁 구현
7. `pipelines/train_3dgs.py` 스텁 구현
8. `tests/test_transform_ops.py`부터 작성
9. `scripts/run_colmap.sh`, `scripts/train_3dgs.sh` 기본 실행 스크립트 작성
10. `docs/architecture.md`에 데이터 흐름 다이어그램 초안 반영

---

## 9) 리스크와 대응

- **리스크 1: 초기 의존성 충돌 (CUDA/PyTorch/NerfStudio)**
  - 대응: 환경 스냅샷(`requirements-lock`, 설치 로그)과 버전 매트릭스 문서화
- **리스크 2: 2D→3D 매핑 불안정**
  - 대응: voting 가중치/가시성 조건을 config로 노출하고 ablation 가능하게 설계
- **리스크 3: 실시간 편집 프레임 드랍**
  - 대응: 편집 중 LOD/해상도 다운샘플 옵션 도입

---

## 10) 의사결정 체크포인트

- 체크포인트 A: Data Acquisition 끝난 시점에 데이터 품질(시점 다양성/블러) 승인
- 체크포인트 B: 3DGS 학습 MVP 결과에서 시각 품질/속도 승인
- 체크포인트 C: ID 매핑 정합도 기준(예: 샘플 씬 정확도) 승인
- 체크포인트 D: 편집 UX(선택 정확도, 조작 반응성) 승인

