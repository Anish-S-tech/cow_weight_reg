# Cattle Weight Estimation System — Project Plan

Status: **awaiting approval — no implementation has started.**

Scope decisions already confirmed with the stakeholder (recorded here for traceability, not
re-litigated below):

| Decision | Answer |
|---|---|
| Training data | No proprietary dataset exists. Use public datasets, but do **not** download them during this build (local storage constraint). Design the data layer to be pluggable so real data can be trained on later, externally. |
| Deployment target | Local Docker Compose only. No cloud provider, no IaC. |
| Product maturity | Treated as a **production product for real users** — auth, persistent multi-user DB, observability, and operational hardening are in scope, even though the shipped ML model will initially be trained only on synthetic data. |
| Stage A (cow detector) license | Ultralytics YOLOv8-seg (AGPL-3.0), accepted for now since this stays self-hosted; revisit before any closed-source resale. |

---

## 1. Project Goal

Build a complete, production-grade system that **estimates a cow's body weight in kilograms from
one or more photographs**, end to end:

- A **machine learning pipeline** (PyTorch) that detects the animal in a photo and regresses its
  body weight, with an honest uncertainty estimate rather than a bare number.
- A **FastAPI backend** that serves predictions to real, authenticated, multiple users, persists
  history, and is hardened the way a real product would be (rate limits, input validation, logging,
  metrics, health checks).
- A **React + Vite frontend** that lets a user log in, take/upload a photo with capture guidance,
  optionally calibrate for scale, and see a result with a confidence range.
- A **Docker Compose deployment** that brings the whole stack up locally with one command.
- **Tests, linting, type checking, and documentation** at every layer, to the standard of what a
  professional ML/backend/frontend team would ship, not a prototype.

The deliverable is judged on engineering quality (architecture, correctness, test coverage,
documentation, honesty about limitations) — not on the trained model's accuracy, since no real
labeled cattle-weight data is used in this phase by explicit decision.

## 2. Technical Challenges

These are the problems that actually make this project hard, in roughly the order they bite:

1. **Monocular scale ambiguity (the core CV problem).** A single 2D photo does not encode absolute
   size — a small cow close to the camera and a large cow far away can produce an identical image.
   Published approaches solve this with depth sensors, 3D point clouds, or a fixed/known capture
   protocol. We have none of those available, so the system must either (a) accept an optional
   calibration input (a reference object of known size, or a known camera distance) to convert pixel
   measurements to real-world centimeters, or (b) fall back to scale-invariant body-shape ratios with
   explicitly wider uncertainty. This is a physics/information limit, not a bug to be coded away, and
   the UI/docs must say so plainly rather than implying false precision.
2. **No labeled training data available in this build.** Public datasets exist (cattle images with
   recorded weights) but downloading them is explicitly out of scope here. The model that ships with
   this repo will therefore be trained on small, procedurally generated synthetic data — enough to
   prove the entire pipeline works mechanically — and must be unambiguously labeled "demo / not
   trained on real cattle" everywhere it surfaces (API response, UI banner, model card). Real accuracy
   is unproven until someone runs the (already-built) training script against real data outside this
   session.
3. **Getting a detector without any training data at all.** Segmenting "where is the cow in this
   photo" normally needs its own labeled dataset. We avoid this by using an object-detection/
   segmentation model pre-trained on COCO, which already has a "cow" class — so Stage A needs zero
   project-specific training data, only a pretrained checkpoint download (a model artifact, not a
   dataset).
4. **Uncertainty, not just a point estimate.** Because of (1) and (2), a bare "412 kg" number would
   overstate confidence. The model must output a predicted range, and the system must visibly degrade
   that confidence when no calibration is supplied, rather than presenting all predictions identically.
5. **"Production" requirements layered on top of an admittedly unproven model.** Real auth, a real
   multi-user database, rate limiting, structured logging, metrics, and health checks all need to be
   built properly, while staying honest in the product surface that the underlying model is a demo.
   These two facts (production-grade infra, demo-grade model) must coexist without the UI/API
   misrepresenting either.
6. **License exposure from a runtime ML dependency.** The chosen detector (Ultralytics YOLOv8) is
   AGPL-3.0, and it runs inside the live backend at request time — not just during training — so the
   license condition actually attaches to the deployed product. The architecture must keep this
   swappable (already decided: BSD-3 Mask R-CNN as a documented alternative behind the same interface).
7. **Keeping training-only dependencies out of the production image where possible.** `ultralytics`/
   `torch`'s full training stack is large; the serving path should depend on as little of it as
   reasonably achievable (a TorchScript-exported model + a thin inference shim) without duplicating
   model code between two independently versioned packages.
8. **Testing an ML pipeline with zero real images.** Unit and integration tests, and even the shipped
   demo checkpoint's training run, must all work against procedurally generated synthetic images, with
   the same code paths real data would later use — this needs a deliberately designed synthetic data
   generator, not an afterthought.
9. **Local dev environment friction.** Target machine is Windows, Python 3.10 is the default
   interpreter (3.12 required), and GNU Make is not installed. Tooling choices (`uv` for Python
   environments/versions, npm scripts as the primary task runner instead of assuming `make` is present)
   need to work without asking the user to install extra system tools first.
10. **Privacy of uploaded photos.** Images may contain EXIF GPS/location metadata and depict
    identifiable property; the backend should strip metadata and avoid retaining raw images by default.

## 3. Required Technologies

**ML / data**
- Python 3.12, PyTorch + torchvision (backbones, Stage A alternative, tensors)
- Ultralytics YOLOv8-seg (Stage A detector, default) — AGPL-3.0, accepted per decision above
- OpenCV (`opencv-python-headless`) — mask geometry (min-area-rect for body length/height)
- Pillow, NumPy — image I/O and array ops, synthetic image generation
- Typer + PyYAML + Pydantic — typed, config-file-driven training CLI

**Backend**
- FastAPI + Pydantic v2 (API layer, request/response validation)
- SQLAlchemy 2.0 (async) + Alembic (persistence + migrations)
- PostgreSQL 16 (database), Redis 7 (rate-limit storage / light caching)
- `passlib` + `argon2-cffi` (password hashing), `PyJWT` (access/refresh tokens)
- `slowapi` (rate limiting), `structlog` (structured JSON logging), `prometheus-client` (metrics)
- `uvicorn` (ASGI server)

**Frontend**
- React 18 + TypeScript + Vite
- TanStack Query (server state/data fetching), Zustand (client/auth state)
- Vitest + React Testing Library + MSW (component/hook tests with mocked network)

**DevOps / tooling**
- Docker + Docker Compose (orchestration), Nginx (frontend static serving + API proxy)
- GitHub Actions (CI: lint + test, no deploy step — local-only scope)
- `uv` (Python dependency/venv/interpreter management — already installed, resolves the
  Python-3.10-by-default problem cleanly)
- Black (formatting), Ruff (linting + import sorting), mypy (type checking), pre-commit
- npm scripts as the cross-platform task runner (works without installing GNU Make on Windows)

## 4. Complete Architecture

### 4.1 ML pipeline — two-stage detect-then-regress

```
Photo ──► Stage A: Detector (frozen, pretrained, zero project-specific training)
            │  YOLOv8-seg (COCO "cow" class) → bbox + mask + confidence
            │  0 or 2+ confident cow detections → reject with a clear error, never guess
            ▼
          Crop + mask → pixel measurements (length/height via min-area-rect, mask-area
          ratio, aspect ratio) via OpenCV
            │
            ├─ optional calibration (reference object cm + px, or known distance) →
            │   pixel-to-cm conversion of the same measurements; absent → sentinel
            │   "no calibration" flag, not just zeros
            ▼
Stage B: Weight Regressor (trained — synthetic now, real data later, externally)
   CNN backbone (EfficientNet-B0, ImageNet-pretrained) on the cropped image
        ⊕  fused with the numeric measurement feature vector via a small MLP head
        ▼
   outputs (mu, log_var) → predicted weight ± a confidence interval, trained with
   Gaussian NLL loss; interval is heuristically widened further when uncalibrated
```

Every checkpoint ships with `metadata.json` (`trained_on`, `is_demo`, feature-schema version) so the
backend can refuse to load an incompatible or undisclosed model rather than silently mis-predicting.

### 4.2 Backend — clean architecture, strict dependency direction

```
api/            FastAPI routers, middleware, DI wiring, pydantic schemas (request/response)
  │  depends on
application/    use cases (PredictWeight, RegisterUser, GetHistory, ...), pure orchestration
  │  depends on
domain/         entities, value objects, abstract ports (Detector, WeightPredictor,
                UserRepository, PredictionRepository) — zero framework imports
  ▲  implemented by
infrastructure/ SQLAlchemy repositories, Argon2/JWT services, the ML adapters (YOLO/Mask R-CNN,
                TorchScript Stage B), Redis client
```

`domain/` never imports FastAPI, SQLAlchemy, or torch — it is pure Python, fully unit-testable with
hand-written fakes. The ML model is consumed via a TorchScript export plus a thin inference shim, so
the backend's serving image does not need to depend on the full training package (`ultralytics`,
training-time `torch`) — only `torch` itself plus the exported `.pt` artifact.

### 4.3 Request flow (a single prediction)

```
Client ──multipart upload──► FastAPI /api/v1/predictions
  1. validate content-type + size (reject before fully buffering large bodies)
  2. decode image (Pillow), strip EXIF (privacy)
  3. run Stage A → 422 NO_COW_DETECTED / MULTIPLE_COWS_DETECTED if not exactly one confident cow
  4. crop + compute pixel measurements
  5. validate + apply calibration if provided, else mark uncalibrated
  6. run Stage B (TorchScript, loaded once at startup) → (mu, log_var)
  7. widen interval if uncalibrated; build response (estimate, interval, is_demo disclaimer)
  8. persist PredictionRecord; return 201 with the result
```

Every domain-level failure (bad image, no cow found, bad calibration, inference error) is a typed
exception mapped to a consistent JSON error body by one exception handler — never a bare 500 leaking
a stack trace to the client (the real trace still goes to structured server-side logs).

### 4.4 Frontend architecture

`PredictPage` (upload + always-visible capture guidance + optional calibration form) → TanStack
Query mutation → `WeightResultCard` (point estimate, confidence-interval bar, a non-dismissible
"demo model" banner whenever `model_info.is_demo` is true, never fine print) → `HistoryPage`
(paginated past predictions). Zustand holds the auth token/session; TanStack Query owns all
server-state fetching/caching/retry so the slow (CPU-bound) predict call gets proper loading/error
UI for free instead of hand-rolled state.

### 4.5 Deployment architecture (Docker Compose)

```
            ┌────────────┐      ┌──────────────┐
  browser ──┤  frontend  │──────┤   backend    │
            │  (nginx)   │ /api │  (FastAPI)   │
            └────────────┘      └──────┬───────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼                            ▼
                    ┌──────────┐                ┌──────────┐
                    │ postgres │                │  redis   │
                    └──────────┘                └──────────┘
```

A one-shot `migrate` service runs `alembic upgrade head` before the backend starts accepting
traffic. All containers run as non-root users. No service is reachable from outside the compose
network except frontend (and backend, on a documented port, for local API debugging).

## 5. Folder Structure

```text
cow_claude/
  PROJECT_PLAN.md                  (this file)
  README.md  ARCHITECTURE.md  MODEL_CARD.md  RUNBOOK.md  CONTRIBUTING.md
  .gitignore  .env.example  .pre-commit-config.yaml  pyproject.toml
  package.json                     root npm scripts = cross-platform task runner (no `make` needed)
  docker-compose.yml
  .github/workflows/ci.yml

  backend/
    Dockerfile  pyproject.toml  alembic.ini
    alembic/{env.py, versions/}
    src/app/
      main.py  config.py  logging_config.py
      domain/{entities, value_objects, repositories, services}
      application/{auth, prediction, dto}
      infrastructure/{db, security, ml, cache, storage}
      api/{deps.py, middleware, v1}
      schemas/
    tests/{unit, integration, fixtures}

  ml/
    pyproject.toml
    src/cattle_ml/
      config/{train_config.py, configs/*.yaml}
      data/{dataset_interface.py, synthetic_dataset.py, kaggle_csv_dataset.py,
            cid_dataset.py, transforms.py, datamodule.py}
      stage_a/{detector_base.py, yolo_seg_detector.py, maskrcnn_detector.py, measurements.py}
      stage_b/{backbone.py, fusion_head.py, model.py, losses.py}
      training/{train.py, trainer.py, export.py}
      evaluation/{metrics.py, evaluate.py}
    scripts/{download_dataset.py (manual-only, never auto-invoked),
             train_demo_checkpoint.py, export_checkpoint_to_backend.py}
    artifacts/demo/                committed: the shipped demo checkpoint + metadata.json
    tests/

  frontend/
    Dockerfile  nginx.conf  package.json  vite.config.ts  tsconfig.json
    src/{routes, components/{upload, result, history, common}, api, hooks, store, types, lib}
    tests/

  docs/
    api/openapi_notes.md
    images/system_overview.mmd

  scripts/
    dev_bootstrap.ps1   (Windows path: provisions uv-managed venvs, installs pre-commit, etc.)
```

Rationale for the two independent Python packages (`backend/`, `ml/`) and `package.json`-as-task-
runner instead of a Makefile is covered in the Risks/decisions already resolved during architecture
review (see Section 7 and the prior approved design notes carried into `ARCHITECTURE.md` once
written).

## 6. Development Roadmap

Each phase ends in a working, tested, documented slice. Work pauses for an explicit go-ahead between
phases — nothing here starts until this document is approved.

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | Repo scaffolding: directory skeleton, root tooling configs, git init, npm-script task runner, pre-commit | `pre-commit run --all-files` and the lint script pass on an empty-but-valid codebase; initial commit exists |
| 2 | ML package core: synthetic data generator, Stage A wrappers, Stage B fusion model, losses/metrics, unit + training-smoke tests | All `ml` tests pass with zero network access; a 2-epoch smoke run shows loss decreasing |
| 3 | Train + export the demo checkpoint | TorchScript checkpoint + `metadata.json` (`is_demo: true`) committed; `evaluate.py` output logged |
| 4 | Backend domain + application layers (no FastAPI/DB yet) | Unit tests cover every use-case branch (incl. no-calibration, no-cow-detected) against hand-written fakes; zero framework imports in `domain/` |
| 5 | Backend infrastructure + API: DB models, Alembic, auth, ML adapters wired to the real checkpoint, routers, health/metrics | Integration tests green against real Postgres; register→login→predict→history works end to end; rate limiting and auth verified by tests |
| 6 | Frontend: all pages, API client, auth flow, predict + calibration UI, history, demo banner | Vitest suite green; manual run against the Phase-5 backend completes the full user journey |
| 7 | Dockerization + Compose | `docker compose up` from a clean checkout works with zero manual steps beyond copying `.env.example`; containers verified non-root |
| 8 | CI, security/hardening pass, final documentation | CI green on a real PR; manual security checklist confirmed; README/ARCHITECTURE/MODEL_CARD/RUNBOOK cross-links resolve |
| 9 (deferred, external) | Real training run against a public dataset (Kaggle/CID), executed by the user outside this session using the already-built training script | Not executed here — documented as a how-to only, per the no-dataset-download constraint |

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Monocular weight estimation has an inherent accuracy ceiling without calibration (literature suggests ±15–25% error uncalibrated) | Users may expect scale-replacing precision and be misled | Mandatory uncertainty output, visible low-confidence flagging, MODEL_CARD.md and UI copy explicitly frame this as decision support, not a scale replacement |
| Shipped model is trained only on synthetic data | Zero guarantee of real-world accuracy | Hard-coded, unmissable "demo model" disclosure in API response + persistent UI banner + model card; real training path fully built and documented, just not executed here |
| Ultralytics' AGPL-3.0 license runs inside the live backend | Copyleft obligations if ever offered as closed-source SaaS | Decision already made and recorded; both detectors sit behind one interface so switching to BSD-3 Mask R-CNN later is a config change, not a rewrite |
| Large ML dependency footprint (`torch`, `torchvision`, `ultralytics`) slows installs/builds | Slower Phase 2 setup and Docker image builds | Two-package split keeps training-only deps out of the backend serving image where feasible; `uv` for materially faster resolves/installs than pip |
| Windows dev machine lacks GNU Make and ships Python 3.10 by default | Tooling instructions that assume `make`/Python 3.12 would silently fail for this user | `uv` manages a real 3.12 interpreter per package regardless of system Python; npm scripts (Node is already installed) replace the Makefile as the cross-platform command surface |
| Uploaded photos may contain EXIF GPS/location data or depict identifiable property | Privacy/safety exposure | EXIF stripped on decode; raw images not retained by default (configurable) |
| Breadth of "production-grade" requirements (auth, multi-user DB, observability, hardening) layered onto a single build | Scope creep, partial implementations, inconsistent quality across phases | Strict phase gates with explicit validation criteria per phase (Section 6); no phase marked done with failing tests or partial features |
| Public dataset metadata (schema, exact license terms, class balance) was only verified via web search snippets, not by inspecting the actual files (deliberately, per the no-download constraint) | The documented `kaggle_csv_dataset.py` / `cid_dataset.py` adapters' assumed CSV schema could be slightly wrong until someone runs Phase 9 for real | Schema is isolated behind the same `CattleWeightDataset` interface as the synthetic generator — fixing a wrong column name later is a one-file change, not a redesign |

## 8. Timeline Estimate

Two different framings, since this is being built by an AI coding agent rather than a human team:

- **If staffed as a human engineering effort**, this is realistically a multi-week solo project for
  an experienced ML/full-stack engineer: ~1 week research + data-layer design, ~1–1.5 weeks ML
  pipeline, ~1.5 weeks backend, ~1 week frontend, ~0.5–1 week DevOps/docs/hardening — **roughly
  5–7 weeks** end to end, or 2–3 weeks for a small team working phases in parallel.
- **As agent-driven build time** (this session/future sessions), each phase is roughly one focused
  working block, dominated by actual command runtime (dependency installs, training runs, Docker
  builds, test suites) rather than typing speed:

  | Phase | Rough active time |
  |---|---|
  | 1 — scaffolding | 15–30 min |
  | 2 — ML core + tests | 45–90 min |
  | 3 — train/export demo checkpoint | 20–40 min |
  | 4 — backend domain/application | 45–60 min |
  | 5 — backend infra/API (largest phase) | 90–120 min |
  | 6 — frontend | 90–120 min |
  | 7 — Docker/Compose | 45–60 min |
  | 8 — CI/hardening/docs | 45–60 min |

  **Total ≈ 7–10 hours of active work**, spread across as many sessions as you want checkpoints for —
  calendar time depends entirely on how quickly each phase gets reviewed and approved.
