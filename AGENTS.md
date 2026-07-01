# AGENTS.md

## Purpose
This repository builds **GSegmenter**, an intelligent 3D object segmentation and interactive editing system based on:
- Segment Anything Model (SAM)
- 3D Gaussian Splatting (3DGS)
- Multi-view semantic lifting from 2D masks to 3D Gaussian entities
- Real-time object manipulation and geometric infilling

Agents working in this repository must preserve the core goal:
1. Lift 2D segmentation signals into consistent 3D object identities.
2. Support interactive object-level editing such as selection, translation, rotation, and deletion.
3. Maintain real-time rendering quality and system stability.
4. Avoid changes that silently break the COLMAP → GS training → identity mapping → editor pipeline.

---

## Product Context
The system is organized around the following conceptual stages:

1. **Data Acquisition**
   - Use COLMAP outputs such as camera poses and sparse point clouds.
   - Any preprocessing code must keep coordinate systems explicit and documented.

2. **GS Training**
   - Training is expected to be compatible with NerfStudio-style Gaussian Splatting workflows.
   - Changes affecting training configuration, data parsing, or checkpoint formats must be backward-conscious.

3. **Identity Mapping**
   - 2D SAM masks are transferred into 3D Gaussian space.
   - Object identity assignment should prioritize consistency across views over per-frame local accuracy.
   - Voting / aggregation logic must be deterministic when possible.

4. **Interactive Editor**
   - Object-level transforms should be implemented as explicit state updates.
   - Transform operations must clearly distinguish:
     - object-local transform
     - world transform
     - Gaussian parameter updates (e.g. means, covariance, rotation-related state)
   - Editing code must favor reversible and debuggable operations.

5. **Geometric Infilling / Scene Repair**
   - When implementing deletion or displacement repair, preserve visual continuity.
   - Prefer local, bounded algorithms before introducing global recomputation.

---

## Agent Priorities
When making decisions, follow this priority order:

1. **Correctness of 3D geometry and object identity**
2. **Interactive responsiveness**
3. **Reproducibility and debuggability**
4. **Code clarity**
5. **Experimental extensibility**

Do not trade correctness for micro-optimizations unless profiling proves the bottleneck.

---

## Environment Assumptions
Target environment typically includes:
- Python 3.10+
- CUDA 11.8 or 12.1
- NVIDIA GPU, preferably RTX 30/40 series

Agents should:
- avoid introducing dependencies that conflict with PyTorch/CUDA compatibility,
- keep installation simple,
- document any extra system requirements immediately in README or module docs.

---

## General Working Rules
- Make the **smallest coherent change** that solves the task.
- Preserve existing public interfaces unless the task explicitly requests refactoring.
- Prefer incremental patches over broad rewrites.
- Do not rename files, classes, or functions unless necessary.
- Do not introduce hidden magic constants; define them near configuration boundaries.
- Keep experimental code isolated and clearly labeled.

If a task is ambiguous, choose the option that best supports the repository’s main pipeline rather than adding unrelated abstractions.

---

## Code Style
### Python
- Follow PEP 8 unless the repository already uses a stricter local convention.
- Use type hints for new or significantly modified functions.
- Keep functions focused; split long functions when logic spans multiple stages.
- Use descriptive names tied to graphics / vision semantics.
  - Good examples:
    - `project_mask_to_gaussians`
    - `aggregate_multiview_votes`
    - `apply_object_transform`
    - `update_gaussian_covariance`
- Avoid vague names like `process_data`, `handle_all`, `tmp_fn`.

### Comments
- Comment the *why*, not the obvious *what*.
- For geometry code, document:
  - coordinate frame assumptions,
  - tensor shapes,
  - unit conventions,
  - numerical stability considerations.

### Logging
- Prefer structured, minimal logs over noisy print statements.
- Log important transitions such as:
  - data load summary,
  - training config,
  - identity mapping statistics,
  - edit operation type and affected object IDs,
  - repair/infilling triggers.

---

## Geometry and Vision Safety Rules
Because this project mixes 2D segmentation, multi-view projection, and 3D Gaussian editing, agents must explicitly guard against:

- coordinate frame mismatches,
- inconsistent camera intrinsics/extrinsics handling,
- silent tensor shape broadcasting bugs,
- object ID instability across views,
- destructive in-place edits that make rollback impossible,
- invalid covariance or transform updates that destabilize rendering.

When changing math-heavy code:
1. State expected input/output shapes.
2. Verify frame convention.
3. Add at least one sanity check or assertion.
4. Avoid silent fallback behavior unless clearly documented.

---

## Performance Rules
Real-time behavior matters.

Agents should:
- prefer batch tensor operations over Python loops in hot paths,
- minimize CPU↔GPU transfers,
- avoid unnecessary memory copies,
- profile before optimizing non-critical code,
- keep debug visualizations optional and gated.

Do not add expensive validation inside per-frame rendering paths unless guarded by a debug flag.

---

## Testing Expectations
For meaningful changes, agents should add or update tests when feasible.

Priority test areas:
1. Projection / backprojection consistency
2. Multi-view identity aggregation
3. Object transform correctness
4. Scene editing invariants
5. Numerical stability of Gaussian parameter updates

At minimum, validate:
- tensor shapes,
- no-NaN / no-inf behavior,
- deterministic behavior under fixed seeds where applicable,
- unchanged behavior for unaffected objects.

If a full automated test is not practical, provide a short reproducible verification procedure in the task output.

---

## File and Module Organization
When adding code:
- place training-related logic near training modules,
- place segmentation lifting logic near mapping / association modules,
- place editor logic near interaction / transform modules,
- place experimental notebooks or scripts outside core runtime paths.

Avoid mixing:
- UI/editor logic with low-level geometry kernels,
- one-off experiments with production inference/training code,
- dataset-specific hacks in generic modules.

If a helper is only used once, prefer keeping it local unless reuse is likely.

---

## Configuration Rules
- Expose tunable parameters through config files, dataclasses, or clear module-level constants.
- Do not hardcode dataset paths, checkpoints, or machine-specific absolute paths.
- New config options must include:
  - sensible default,
  - short explanation,
  - impact on performance/quality if non-obvious.

---

## Documentation Rules
Any agent changing behavior in core pipeline areas must update documentation for:
- required inputs,
- expected outputs,
- config changes,
- new runtime assumptions,
- migration notes if behavior changed.

For math-heavy functions, include brief docstrings with:
- purpose,
- key arguments,
- tensor/frame expectations,
- returned values.

---

## Forbidden Changes
Agents must not:
- replace core pipeline components with unrelated frameworks without explicit request,
- remove reproducibility hooks or configuration surfaces,
- introduce closed-source or non-redistributable dependencies by default,
- hardcode local filesystem paths,
- suppress failures that should be visible to developers,
- merge unrelated refactors into a task-specific patch.

---

## Preferred Task Output
When completing a coding task in this repository, provide:
1. What changed
2. Why it was needed
3. Any assumptions
4. How to verify it
5. Risks or follow-up items

Keep the report concise and technical.

---

## Review Checklist
Before finalizing changes, agents should verify:

- Does this preserve the SAM → 3D identity lifting workflow?
- Does this preserve or improve editing correctness?
- Are coordinate frames documented and consistent?
- Are tensor shapes and device placement safe?
- Does this avoid harming real-time performance?
- Are new dependencies justified?
- Is the change reproducible and debuggable?

If any answer is “no” or “unclear”, revise before submitting.