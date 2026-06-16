# State of the Art: Computer-Vision-Based Cattle Weight Estimation

Literature review conducted via search across recent (2019–2026) peer-reviewed papers, preprints,
and systematic reviews. No datasets were downloaded as part of this research — only published
papers, abstracts, and review articles were consulted (consistent with this project's
no-dataset-download constraint).

Anchor reviews used to triangulate the field's overall shape:
- *Computer vision-based weight estimation of livestock: a systematic literature review* (Taylor &
  Francis, 2021)
- *Computer Vision-Based Measurement Techniques for Livestock Body Dimension and Weight: A Review*
  (MDPI Animals, 2024)
- *Recent advances in computer vision for non-contact phenotyping and weight estimation in
  livestock: a systematic review* (ScienceDirect, 2025)
- *Computer vision-based livestock weight measurement: principles, methods, and challenges*
  (OAE Publish, 2026)
- *Analyzing Data Modalities for Cattle Weight Estimation Using Deep Learning Models* (MDPI/PMC,
  2024) — directly compares RGB, depth, RGB+depth, segmentation, and segmentation+depth inputs

## 1. Quick-Reference Comparison

| Approach | Advantages | Disadvantages | Compute | Data needs | Expected accuracy |
|---|---|---|---|---|---|
| Regression CNN (direct image→weight) | Simplest pipeline, end-to-end, no intermediate annotations beyond weight labels | Fully exposed to monocular scale ambiguity; accuracy collapses if capture distance/angle isn't consistent | Low–moderate (ResNet/VGG-class backbone, single forward pass) | Paired image+weight only, but needs *consistent capture protocol* to substitute for missing scale cues — typically 500–2000+ images | Weak-to-moderate without a fixed protocol; literature on uncalibrated single-RGB approaches commonly reports roughly 10–25% MAPE |
| EfficientNet (as backbone) | Best accuracy-per-FLOP of the CNN family; compound scaling gives a size/accuracy dial (B0→B7); strong ImageNet transfer learning track record; practical for edge/CPU | Same scale-ambiguity ceiling as any 2D-RGB-only backbone; gains are about efficiency, not solving the core problem | Low (B0: ~5M params) to moderate (B4+); CPU-inference-friendly at B0–B2 | Same as regression CNN; benefits more from transfer learning so tolerates smaller fine-tuning sets (low hundreds) better than training from scratch | Comparable to other CNN regression approaches at equal input modality; mainly an efficiency/deployability win, not an accuracy ceiling-breaker |
| ConvNeXt | Modernized CNN (large kernels, LayerNorm/GELU, ViT-inspired macro design) approaching transformer-level accuracy while keeping CNN data efficiency; reported as a strong backbone for *depth-image* cattle weight prediction and in dual-stream RGB-D pig weight networks (ConvNeXtV2) | Larger and slower than EfficientNet-B0/B1 for comparable accuracy on small livestock datasets; less battle-tested in the cattle literature specifically (most citations are recent, 2024–2025, and pig/depth-focused) | Moderate–high (more params/FLOPs than EfficientNet at a given accuracy tier) | Benefits from more data than EfficientNet to fully exploit its larger capacity; works well as a depth/point-cloud feature extractor when paired with 3D data | Promising, slightly ahead of EfficientNet-class CNNs in the few studies that compare them directly, especially in depth-input settings — but evidence base is thinner |
| Vision Transformers (ViT) | Captures global, whole-body shape context (not just local texture); SOTA on large general vision benchmarks; emerging cross-attention RGB-D fusion variants (He et al.) and zero-shot segmentation (SAM2) / zero-shot depth (Depth Pro) use ViT encoders effectively | Notoriously data-hungry; literature explicitly notes "limited adoption in animal body weight prediction due to small dataset size and lack of pretrained weights tailored to depth images in livestock"; quadratic attention cost on high-resolution images | High (full ViT fine-tuning) — though usable cheaply when only used as a *frozen* zero-shot component (SAM2, Depth Pro) rather than trained from scratch | Needs the most data of any option here to fine-tune well; the field's typical livestock datasets (hundreds to ~20k images) are on the small end for ViT fine-tuning, so most current use is as a frozen, pretrained zero-shot tool, not a trained regression head | Strong when used as a frozen zero-shot tool (segmentation, depth); unproven/inconsistent when fine-tuned end-to-end for weight regression on typical livestock dataset sizes |
| YOLO-based methods (detection / keypoint) | Fast, lightweight, real-time-capable, runs well on CPU/edge; mature ecosystem; in this domain mostly used for keypoint detection (e.g., SCW-YOLO, YOLOv8-DMC) feeding 11+ body measurements (height, chest depth/width, girth, etc.), or for calf weight from depth images with threshold segmentation | YOLO itself only locates/measures — weight accuracy depends entirely on the downstream measurement→weight regression step, not YOLO; keypoint accuracy degrades under occlusion/poor lighting without extra attention modules (YOLOv8-DMC adds three just to compensate) | Low (YOLOv8n/s-class models, tens of ms per image on CPU) | Needs *keypoint-annotated* images (more annotation effort per image than a bare weight label), but dataset sizes used in practice are modest (low thousands) | Good as a measurement-extraction front end; end-to-end weight accuracy is determined by the regression stage built on top, typically competitive with or better than plain CNN regression because explicit body measurements are more weight-correlated than raw pixels |
| Segmentation-based methods (Mask R-CNN, etc.) | Per-pixel mask gives strong silhouette features (area, convex-hull, elongation, contour length) that correlate well with weight; some of the highest reported accuracies in the field; works from a single smartphone photo (PickAMoo: 567 annotated images, F1=0.98 for the segmentation step itself); pairs naturally with depth for top-tier results | Heavier model than plain detection (full instance-segmentation network); accuracy figures at the very top of the range (R²=0.96–0.98) come from controlled/top-view or LiDAR-assisted setups, not casual side-on phone photos | Moderate (Mask R-CNN ResNet50-FPN-class) to low (lightweight seg variants, e.g. YOLO-seg) | Needs pixel-mask annotations (heaviest per-image annotation cost of any method here) but a few hundred to ~1-2k well-annotated images is enough given transfer learning, per PickAMoo and the original Mask R-CNN cattle-contour paper (2019) | Among the best reported in the literature: R²=0.91–0.98 and MAPE as low as ~2% in studies that pair segmentation with depth or a controlled top-view; side-view-only, no-depth setups land lower (R²≈0.91, RMSE≈27kg in one cited study) |
| Multi-view / multi-camera estimation | True 3D geometry (not a 2D proxy) — most physically grounded measurements; reported sub-2cm uncertainty in body width/length; fast capture (~0.08s in one gantry system) | Highest infrastructure cost: multiple synchronized/calibrated cameras, a fixed station or gantry, registration/3D-reconstruction software; real-world failure modes include occlusion by fencing and camera vibration requiring extra noise-removal steps; fundamentally incompatible with "one phone photo" product use cases | High (multi-camera capture + point-cloud registration + 3D reconstruction pipeline; typically a dedicated workstation/server, not a phone or thin client) | Needs a fixed, calibrated capture rig rather than more images per se — data collection is operationally heavy (physical installation) even if labeled samples needed can be moderate | Generally the most accurate geometry-grounded category (sub-2cm dimensional uncertainty reported), but is reported alongside, not as a replacement for, a final ML regression step from the recovered measurements to weight |
| Depth estimation (hardware RGB-D/LiDAR or monocular zero-shot) | Directly attacks the field's core problem (scale ambiguity) by injecting real metric information; hardware depth (Kinect/RealSense/ToF/LiDAR) gives some of the best reported accuracies (R²=0.91–0.98, MAPE~2–9%); monocular zero-shot depth (Depth Pro, SAM2-adjacent ViT-based models) is now emerging as a **hardware-free** way to get an approximate metric signal from an ordinary photo, and is explicitly called out in the 2024 modality-comparison literature as a promising direction | Hardware depth sensors aren't available to a typical phone-only user (breaks the "any photo" product goal); monocular zero-shot depth is general-purpose (trained on everyday scenes/objects), not livestock-specific, so its metric accuracy on animal bodies/fur is not yet well validated in this domain; adds a second heavy model to the inference pipeline | Hardware depth: low at inference (sensor does the work) but requires the device; Monocular zero-shot depth: moderate–high (ViT-based depth models are not small) | Hardware depth: training data can be smaller since depth removes most scale ambiguity; Monocular depth: needs no livestock-specific training data at all if used zero-shot (frozen), only validation data to confirm it transfers reasonably to this domain | Best-in-class when hardware depth is available; monocular zero-shot depth's accuracy *as a calibration signal* for livestock specifically is still an open, actively-researched question rather than an established number |

## 2. Discussion Per Category

### 2.1 Regression CNNs (direct image → weight)
The most basic architecture: a CNN backbone (often ResNet or a generic VGG-style net in older
papers) with its classification head replaced by a single regression output, trained end-to-end on
(image, weight) pairs. Historically the first approach the field tried, and still a fair baseline.
Its core weakness is structural, not a training problem: a 2D RGB image alone does not encode
absolute scale, so the network can only learn weight cues that happen to correlate with apparent
size *given the dataset's incidental capture conditions* (e.g., if every photo in training happened
to be taken from roughly the same distance). This means accuracy is highly sensitive to whether
the deployment capture protocol matches training — a real risk for a consumer product where users
take photos from varying distances. Cheapest to build and explain, but the weakest accuracy ceiling
of anything in this table unless paired with a fixed protocol.

### 2.2 EfficientNet
Not a different *kind* of method so much as a better backbone for the same regression-CNN idea.
EfficientNet's compound scaling (depth/width/resolution scaled together) gives materially better
accuracy-per-FLOP than ResNet-class nets, and its small variants (B0/B1) are practical on CPU —
relevant for a Docker-Compose, possibly-no-GPU deployment. It inherits the same scale-ambiguity
ceiling as any RGB-only backbone (Section 2.1), so its real contribution here is efficiency and
transfer-learning sample efficiency (useful when only hundreds of fine-tuning images are available),
not solving the underlying physics problem.

### 2.3 ConvNeXt
A 2022-era redesign of plain CNNs borrowing macro-design ideas from ViT (large kernels, LayerNorm,
GELU, fewer activation functions) while keeping convolution's data efficiency. In the livestock
literature it shows up specifically as a strong backbone for *depth-image* and RGB-D fusion models
(e.g., a ConvNeXt backbone described as "approaching transformer-level accuracy" for depth-based
cattle weight prediction, and ConvNeXtV2 in a dual-stream RGB-D pig-weight network), rather than as
a plain-RGB regression backbone. It is larger and slower than EfficientNet-B0/B1 for a given small
livestock dataset, and the evidence base specific to cattle is thinner and more recent (mostly
2024–2025) than for EfficientNet or Mask R-CNN. A reasonable "upgrade path" backbone once a real,
reasonably sized dataset (low thousands+) is available, not a clear win on a small or synthetic
dataset today.

### 2.4 Vision Transformers (ViT)
Two very different use cases get conflated under "ViT for this problem": (a) fine-tuning a ViT
end-to-end as the weight-regression backbone, and (b) using a *frozen, pretrained* ViT-based model
zero-shot for a sub-task (segmentation via SAM2, monocular depth via Depth Pro, or feature
extraction). Use case (a) is explicitly reported in the literature as under-adopted for this problem
because livestock datasets are small and there are no livestock/depth-specific ViT pretrained
weights — ViTs need more data than is typically available here to outperform a well-tuned CNN. Use
case (b) is genuinely promising and increasingly common: a frozen ViT-based zero-shot segmenter or
depth estimator can be dropped into a pipeline with **zero additional training data**, exactly the
kind of asset this project already relies on for Stage A (COCO-pretrained zero-shot cow detection).

### 2.5 YOLO-based methods
In this literature, YOLO is almost never the thing that predicts weight directly — it is the fast
front-end that finds the animal and/or anatomical keypoints (e.g., SCW-YOLO, YOLOv8-DMC), which then
feed a separate measurement-to-weight regression stage. This split is a deliberate, sensible
engineering pattern: detection/keypoint location is a well-solved, fast, CPU-friendly problem, so
spending model capacity there is efficient, while the harder weight-regression problem gets a
purpose-built head. Recent variants add attention modules specifically to recover keypoint accuracy
under occlusion and variable lighting (YOLOv8-DMC's three attention blocks), reflecting that
real-world farm photos are messier than lab/chute photos. Strong fit for a product that needs
fast, cheap, CPU-capable inference.

### 2.6 Segmentation-based methods (Mask R-CNN and similar)
Going one level beyond a bounding box, instance segmentation gives a pixel-accurate silhouette, from
which area, convex-hull area, elongation, and contour length can be computed — all measurably
correlated with body weight. This category contains some of the literature's best reported numbers
(R²=0.96–0.98, MAPE as low as ~2% in PickAMoo and related work), though the very best figures come
from controlled top-view or LiDAR-assisted captures rather than arbitrary side-on phone photos
(side-view-only, no-depth results land lower, e.g., R²≈0.91, RMSE≈27kg in one cited comparison).
Annotation cost (pixel masks) is the highest of any 2D method, but transfer learning means a few
hundred to ~1–2k well-annotated images is reportedly enough (PickAMoo used 567).

### 2.7 Multi-view / multi-camera estimation
The most physically rigorous category: multiple synchronized (often depth) cameras reconstruct a 3D
mesh or point cloud, from which true geometric measurements are taken (not 2D proxies). Reported
precision is excellent (sub-2cm dimensional uncertainty, sub-0.1s capture), but the cost is a fixed,
calibrated multi-camera rig — a barn-mounted gantry or station, not something a farmer points a
phone at. Real deployments report real problems (fence occlusion, camera vibration) needing dedicated
mitigation. This is the right category for a fixed weighing/monitoring station product, and the
wrong category for a "snap a photo with your phone" product — which is this project's stated goal.

### 2.8 Depth estimation
Two distinct sub-approaches: **hardware depth** (structured light/ToF sensors like Kinect/RealSense,
or LiDAR) gives a direct, reliable metric signal and underlies many of the field's best accuracy
numbers — but requires hardware most users don't have in a phone-only product. **Monocular zero-shot
depth estimation** (ViT-based models like Depth Pro, or related general-purpose depth models) infers
an approximate depth map from an ordinary RGB photo with no special hardware and no livestock-specific
training data, and the 2024 modality-comparison literature explicitly flags this as a promising,
recent direction. Its main open question for this domain is validated accuracy on animal bodies
specifically (fur, irregular geometry) rather than the general scenes/objects it was trained on — it
is a calibration *signal*, not yet a proven livestock metrology tool.

## 3. Cross-Cutting Findings

1. **Backbone choice (CNN vs. ConvNeXt vs. ViT) matters far less than whether the pipeline has any
   metric scale signal at all.** The accuracy gap between "best 2D-only regression CNN" and "best
   depth/segmentation-assisted method" (often 2–3x in MAPE terms) dwarfs the gap between, say,
   EfficientNet and ConvNeXt at the same input modality. Solving scale ambiguity dominates backbone
   selection.
2. **Modality fusion consistently beats single-modality input.** Every comparative study found
   (RGB+depth, segmentation+depth, keypoints+image, image+height/length in SCA-ConvNet) outperforms
   its single-modality ablation. This favors an architecture that fuses an appearance embedding with
   explicit numeric measurements over either alone.
3. **Datasets in this field are uniformly small by deep-learning standards** — hundreds to ~20k
   images is typical, even in 2024–2026 papers. This favors transfer learning (EfficientNet, frozen
   zero-shot ViT components) over architectures that need large in-domain data to shine (fine-tuned
   ViT, very large ConvNeXt variants).
4. **The practical constraint is almost always hardware availability, not algorithm sophistication.**
   The highest-accuracy methods (multi-view rigs, LiDAR-assisted segmentation) require fixed
   installations or special sensors. Once a product commits to "any phone photo," it has already
   capped its accuracy ceiling below what a barn-mounted depth-camera gantry could achieve — that
   trade-off should be made consciously, not discovered after the fact.
5. **Detection/keypoint/segmentation front-ends + a separate regression stage is the dominant modern
   pattern**, not single-network end-to-end pixel-to-weight regression. This is consistent with
   (and validates) a two-stage detect/measure-then-regress design.

## 4. Recommendation

**For this project's specific constraints — consumer phone photos, no special hardware required,
an optional user-supplied calibration object, and a small/synthetic dataset today with real data
planned later — the literature supports a segmentation/keypoint-driven measurement extraction stage
feeding a CNN-appearance + numeric-measurement fusion regression stage, with monocular depth as a
near-term upgrade rather than a v1 requirement.** Concretely, in the project's own terms:

- **Stage A (detect/segment the animal): keep YOLO-seg.** Section 2.5/2.6 show keypoint/segmentation
  front-ends are the dominant, well-validated pattern, and a zero-shot COCO-pretrained model needs no
  project-specific training data — directly matching the no-dataset constraint.
- **Feature extraction: combine segmentation *and* keypoint-style measurements**, not bounding-box
  geometry alone. The mask gives area/elongation/contour features (Section 2.6's best-performing
  ingredient); min-area-rect/keypoint-style length and height add the explicit body-measurement
  features the keypoint literature (Section 2.5) shows are weight-predictive. The plan's existing
  fusion-head design (CNN embedding ⊕ measurement vector) already reflects finding #2 above
  (modality fusion beats single modality) and should be kept.
- **Stage B backbone: EfficientNet-B0 now; ConvNeXt as a documented upgrade once real data exists.**
  Given a small/synthetic dataset today, EfficientNet's transfer-learning sample efficiency and CPU
  friendliness outweigh ConvNeXt's slightly higher (but thinly evidenced, in cattle specifically)
  ceiling. Re-evaluate ConvNeXt once training on the real Kaggle/CID data (already planned as Phase 9)
  is actually run — that is exactly the point at which ConvNeXt's larger-data advantage could pay off.
- **Do not fine-tune a ViT end-to-end for the regression head at this stage** — the field's own
  finding (Section 2.4) is that current livestock dataset sizes don't favor it. Continue using
  ViT-family models only where they're already used zero-shot/frozen (this project's Stage A
  detector category is open to it; e.g., SAM2 would be a documented BSD-license-friendlier-than-AGPL
  alternative to YOLO-seg worth a future look, separate from the regression head question).
- **Treat monocular zero-shot depth estimation as the highest-value near-term research spike**, ahead
  of architecture changes. It directly attacks the project's hardest acknowledged problem (scale
  ambiguity, RESEARCH.md §2.8 and PROJECT_PLAN.md's Risk #1) without requiring hardware the target
  user doesn't have, and the literature explicitly flags it as an emerging direction rather than a
  solved one — meaning validating it against this project's own held-out data would be a genuine,
  citable contribution, not just plumbing.
- **Do not pursue multi-view/multi-camera rigs** for this product. It is the most accurate category
  in the literature but structurally incompatible with "one phone photo," and would only make sense
  if the product pivoted to a fixed farm-station deployment model.

This recommendation **confirms the architecture already approved in PROJECT_PLAN.md** (YOLO-seg →
measurements → EfficientNet-B0 fusion regression → calibration-aware uncertainty output) as a
literature-consistent v1, and adds two concrete, evidence-based follow-ups for the roadmap: (a)
re-evaluate a ConvNeXt backbone once real training data is used (Phase 9), and (b) spike monocular
zero-shot depth estimation as a software-only calibration enhancement, most naturally as a v2 item
after the v1 (calibration-object-based) approach is shipped and validated.

## 5. Sources

- [Computer vision-based weight estimation of livestock: a systematic literature review](https://www.tandfonline.com/doi/full/10.1080/00288233.2021.1876107)
- [Computer Vision-Based Measurement Techniques for Livestock Body Dimension and Weight: A Review](https://www.mdpi.com/2077-0472/14/2/306)
- [Recent advances in computer vision for non-contact phenotyping and weight estimation in livestock: a systematic review](https://www.sciencedirect.com/science/article/pii/S2214317325000629)
- [Computer vision-based livestock weight measurement: principles, methods, and challenges](https://www.oaepublish.com/articles/ir.2026.11)
- [Analyzing Data Modalities for Cattle Weight Estimation Using Deep Learning Models](https://www.mdpi.com/2313-433X/10/3/72) / [PMC mirror](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10971323/)
- [Evaluating transfer learning strategies for improving dairy cattle body weight prediction in small farms using depth-image and point-cloud data](https://arxiv.org/pdf/2601.01044)
- [Body weight estimation of beef cattle with 3D deep learning model: PointNet++](https://www.sciencedirect.com/science/article/abs/pii/S0168169923005720)
- [Cattle weight estimation using 2D side-view images and estimated depth-based 3D modeling](https://www.sciencedirect.com/science/article/pii/S2772375525003326)
- [SAM 2-Assisted Vision Transformer and Morphometric Feature Engineering for Pig Weight Estimation from RGB Images](https://doi.org/10.3390/app16115708)
- [Industry-scale prediction of video-derived pig body weight using efficient convolutional neural networks and vision transformers](https://www.sciencedirect.com/science/article/pii/S1537511025001795)
- [A unified Vision Transformer (ViT) backbone with Penalty Outside Point Loss for monocular body measurement of Binglangjiang buffaloes](https://www.sciencedirect.com/science/article/pii/S0168169925012724)
- [Estimation of cattle weight from composite image/height/length data with spatial and channel attention convolution network (SCA-ConvNet)](https://link.springer.com/article/10.1007/s11760-024-03398-5)
- [A Review of Visual Estimation Research on Live Pig Weight](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548296/)
- [A Lightweight Automatic Cattle Body Measurement Method Based on Keypoint Detection](https://www.mdpi.com/2073-8994/17/11/1926)
- [YOLOv8-DMC: Enabling Non-Contact 3D Cattle Body Measurement via Enhanced Keypoint Detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12466762/)
- [Predicting Dairy Calf Body Weight from Depth Images Using Deep Learning (YOLOv8) and Threshold Segmentation](https://arxiv.org/pdf/2504.17943)
- [Can 3D point cloud data improve automated body condition score prediction in dairy cattle?](https://arxiv.org/pdf/2601.22522)
- [Korean Cattle 3D Reconstruction from Multi-View 3D-Camera System in Real Environment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11154310/)
- [Multi-view real-time acquisition and 3D reconstruction of point clouds for beef cattle](https://www.sciencedirect.com/science/article/abs/pii/S0168169922003040)
- [Estimating body weight of individual beef heifers using point-cloud reconstruction and machine learning](https://plf.tennessee.edu/wp-content/uploads/sites/229/2024/01/Estimating-body-weight-of-individual-beef-heifers-using-point-cloud-reconstruction-and-machine-learning.pdf)
- [PickAMoo: LIDAR-enhanced mask R-CNN segmentation for precision weight estimation in dairy cattle using smartphone imaging](https://www.nature.com/articles/s41598-026-54742-3)
- [Cattle segmentation and contour extraction based on Mask R-CNN for precision livestock farming](https://www.sciencedirect.com/science/article/abs/pii/S0168169919304077)
- [Depth video data-enabled predictions of longitudinal dairy cow body weight using thresholding and Mask R-CNN algorithms](https://www.sciencedirect.com/science/article/pii/S2772375523001818)
- [CattleNet-XAI: An explainable CNN framework for efficient cattle weight estimation](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0336434)
- [Live Weight Prediction of Cattle Based on Deep Regression of RGB-D Images](https://www.researchgate.net/publication/365072116_Live_Weight_Prediction_of_Cattle_Based_on_Deep_Regression_of_RGB-D_Images)
- [Cattle weight estimation model through readily photos (LaWE)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624021353)
