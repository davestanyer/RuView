# RuView CSI Reference Triage — temporal occupancy, benchmarks, and useful external work

| Field | Value |
|-------|-------|
| Date | 2026-07-05 |
| Status | Research note / implementation seed |
| Scope | Quick triage of two external WiFi-CSI resources Dave shared, with RuView-specific takeaways |
| Primary question | What is useful for RuView right now, especially desk occupancy / false presence / temporal classification? |
| Citation discipline | Treat external repository claims as pointers until reproduced on RuView data. Do not cite reported accuracies as RuView evidence. |

## Executive judgement

Both resources are useful, but in different ways:

1. [`ludlows/CSI-Activity-Recognition`](https://github.com/ludlows/CSI-Activity-Recognition) is a small, old TensorFlow 2 example. It is **not worth importing**, but it is a clean reminder that CSI activity recognition usually works on **sliding temporal windows**, not isolated frames.
2. [`NTUMARS/Awesome-WiFi-CSI-Sensing`](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing) is more valuable: it is a curated map of benchmarks, datasets, model repos, occupancy papers, cross-environment work, and CSI tooling. It should guide the next research spike.

The product-shaped conclusion for RuView:

> RuView should stop asking one CSI frame “is Dave seated / is the room occupied?” and instead classify a **label-pure rolling sequence**, then pass that through a conservative decision layer that is allowed to say `uncertain` rather than hallucinating presence or vitals.

This aligns with current observed RuView failure modes: stable empty-office RF motion can look person-like, while broad per-frame adaptive outputs can disagree with product truth. More undifferentiated CSI-only retraining is likely lower leverage than cleaner temporal windows with ground-truth labels and held-out-recording evaluation.

---

## Current RuView baseline this note is reacting against

The live adaptive classifier in `v2/crates/wifi-densepose-sensing-server/src/adaptive_classifier.rs` is deliberately lightweight:

- 15 engineered features per frame:
  - server features such as variance, motion-band power, breathing-band power, spectral power, dominant frequency, change points, mean RSSI
  - subcarrier amplitude statistics such as mean, std, skew, kurtosis, IQR, entropy, max, range
- filename-discovered classes from `train_*.jsonl`
- per-class statistics plus multinomial logistic regression trained by SGD
- no learned sequence model
- no explicit label-purity windowing
- no recording/session-level validation split enforced by the trainer

That is good for local adaptability and inspectability, but weak for states that are inherently temporal:

- desk-seated vs standing-near-desk
- present-still vs stable empty-office multipath
- transitions into / out of the desk
- temporary node bursts, stale node state, or RF interference
- empty-room false positives where vitals should be null

RuView already has a product-side desk-occupancy memory / conservative decision layer. The next model-layer improvement should feed that layer with better temporal evidence, not bypass it with a single high-confidence per-frame class.

---

## Reference 1 — ludlows/CSI-Activity-Recognition

Source: <https://github.com/ludlows/CSI-Activity-Recognition>

### What it contains

A compact TensorFlow 2 implementation for human activity recognition from CSI amplitude:

- input: CSV CSI amplitude, 90 features per frame
- labels: `bed`, `fall`, `pickup`, `run`, `sitdown`, `standup`, `walk`
- preprocessing:
  - sliding windows of 1000 frames
  - step size 200
  - downsample factor 2
  - keep a window only if activity is present in at least 60% of frames
- model:
  - Bidirectional LSTM
  - attention layer over time
  - dense softmax classifier

### Useful ideas for RuView

#### 1. Classify windows, not frames

The most important takeaway is architectural, not code-level. The repo treats activity as a temporal sequence. RuView’s current adaptive classifier treats each JSONL frame as a sample.

For RuView, the equivalent should be:

- build rolling windows per node and/or per fused stream
- preserve node and recording boundaries
- classify the window state
- expose the window duration, stability, and vote distribution to the API/UI

#### 2. Label-purity gating

The 60% activity-present threshold is directly useful. It avoids training on transition windows whose label is ambiguous.

RuView should use this idea with Dave/camera/manual ground truth:

- keep a training window only if one label dominates the window
- discard mixed windows around sit/stand/leave transitions
- prefer fewer clean windows over many contaminated frames
- evaluate transition windows separately as a product UX problem, not as ordinary training samples

Suggested starting thresholds:

- `>= 0.80` label purity for `desk_seated`, `absent`, `away_from_desk`
- `>= 0.60` only for inherently dynamic labels such as `present_moving`
- below threshold: train/evaluate as `transition_or_uncertain`, not forced into a hard class

#### 3. Attention over time

Attention is useful because not all CSI frames in a window carry equal signal. The key moments may be micro-motion, chair movement, torso settling, or a brief interference burst.

RuView does not need to import their TensorFlow layer, but a future tiny GRU/TCN/attention model should be allowed to weight subsegments of a window.

#### 4. Overlapping windows

The default 1000-frame window / 200-frame stride pattern is not directly transferable, but the shape is right: long enough to contain behaviour, short enough to update product state.

For RuView, start with product-oriented windows:

- 5–15 seconds of CSI history
- 1–3 second stride
- per-node sequences first, fused summary second
- keep enough history for temporal warm-up before judging live state

### What not to copy

Do not copy the code directly:

- TensorFlow 2, old and small
- assumes 90 amplitude features, likely Intel-5300-style data
- no node geometry
- no ESP32-specific schema
- no phase/geometry/fusion handling
- labels are generic HAR, not RuView product states
- likely optimistic validation if overlapping windows are randomly split

### RuView action from this reference

Use it as a design seed for an offline temporal-classifier spike:

1. Convert RuView JSONL to per-node window tensors.
2. Add label-purity gating from camera/manual labels.
3. Split train/test by recording/session, never random neighbouring windows.
4. Train a boring baseline first.
5. Only then test a small GRU/TCN/attention model.

---

## Reference 2 — NTUMARS/Awesome-WiFi-CSI-Sensing

Source: <https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing>

### What it contains

A curated index of WiFi-CSI sensing resources:

- benchmarks
- surveys
- occupancy detection
- human activity recognition
- cross-environment sensing
- multi-modal sensing
- platforms / CSI tools
- datasets
- paper-linked code repos

This is more useful than the ludlows repo because it points to the wider ecosystem and the codebases/datasets worth inspecting next.

### Highest-value items for RuView

#### 1. SenseFi / WiFi-CSI-Sensing-Benchmark

Source linked from the list: <https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark>

Why it matters:

- benchmark-oriented rather than a one-off demo
- PyTorch model zoo for WiFi CSI sensing
- includes NTU-Fi dataset references
- likely has cleaner preprocessing/evaluation patterns

RuView use:

- inspect model/data abstractions
- borrow benchmark harness ideas
- compare RuView JSONL windows against common CSI model input expectations
- avoid copying accuracy claims until reproduced on RuView data

#### 2. NTU-Fi dataset

Listed under datasets via SenseFi.

Why it matters:

- uses Atheros CSI with 114 subcarriers
- contains activities and gait patterns
- richer than legacy 30-subcarrier Intel 5300 datasets

RuView use:

- useful for model experimentation and sanity checks
- not production calibration data
- not a substitute for Dave’s room / ESP32 / node-geometry-specific captures

#### 3. Occupancy detection literature

The list’s occupancy section is more product-relevant than generic HAR.

Especially useful themes:

- Time-Selective RNN for multi-room presence
- self-adaptive occupancy monitoring
- passive motion detection with commodity IoT devices
- FreeDetector / device-free occupancy baselines

RuView use:

- false-positive suppression
- empty-room baselines
- time-selective sequence models
- decision-layer uncertainty and null-vitals semantics

This is the section most aligned with RuView’s immediate office problem: truthful occupancy and desk state, not seven-way activity classification.

#### 4. Cross-environment WiFi sensing

Relevant listed topics/repos include:

- SHARP / SHARPax
- CrossSense
- environment-independent HAR
- adversarial/domain adaptation papers
- AutoFi-style geometric self-supervised learning

RuView use:

- frame calibration as a drift-management problem
- treat room/node/furniture changes as domain shifts
- prefer small per-room adapters and ground-truth refreshes over pretending one universal model will stay true
- validate on held-out days / placements / recordings

#### 5. Multi-modal sensing

Relevant listed resources include:

- MM-Fi: CSI + RGB-D + LiDAR + mmWave
- WiFi + vision multimodal learning
- WiFi pose papers such as Person-in-WiFi / MetaFi lineages

RuView use:

- validates using the SER8 webcam as a temporary calibration oracle
- camera labels should supervise CSI, not become the product dependency
- paired CSI + camera labels should train conservative desk/away/absent discriminators

#### 6. ESP32 CSI Tool

Source linked from the list: <https://stevenmhernandez.github.io/ESP32-CSI-Tool/>

RuView already has its own firmware/runtime path, but this remains useful for sanity-checking:

- ESP32 CSI packet conventions
- amplitude/phase limitations
- known collection practices
- what other ESP32 CSI projects expose or omit

---

## Specific RuView recommendation

### Build a small offline temporal-occupancy spike

Do not start with a deep model. Start with a reliable data/evaluation harness.

Proposed artifact:

`v2/crates/wifi-densepose-sensing-server` remains unchanged initially. Add an offline research script under a research/benchmark path, for example:

- `benchmarks/desk-occupancy-temporal/`
- or `tools/research/desk_occupancy_temporal/`

Minimum spike shape:

1. **Input**
   - RuView JSONL recordings
   - camera/manual labels aligned to timestamps where available
   - node geometry metadata from `data/calibration/`

2. **Window builder**
   - per-node windows
   - fused/global windows
   - no window crosses recording boundaries
   - no window concatenates different node IDs
   - configurable window duration and stride

3. **Label-purity gate**
   - `desk_seated`
   - `standing_near_desk`
   - `away_from_desk`
   - `absent`
   - `transition_or_uncertain`

4. **Baselines first**
   - logistic regression / linear SVM on temporal summary features
   - random forest / gradient boosting if needed
   - only then GRU/TCN/attention

5. **Validation discipline**
   - split by recording/session/day
   - no random split of overlapping windows
   - always report confusion matrix
   - always report false-positive rate for `absent -> present/seated`
   - always report product metrics:
     - `desk_seated` precision
     - empty-room false positive rate
     - null-vitals correctness when absent
     - time-to-stable-state after transitions

6. **Deployment rule**
   - model output is advisory evidence
   - conservative decision layer remains authoritative
   - high `model_seated` cannot force product `desk_seated` without supporting evidence
   - ambiguous windows become `uncertain`, not fake certainty

---

## Anti-patterns to avoid

1. **Do not import public CSI models and expect direct gains.**
   Most public datasets use Intel 5300, Atheros, Nexmon, or router CSI schemas. RuView’s ESP32 multistatic setup is different.

2. **Do not random-split overlapping windows.**
   It leaks near-identical neighbours into train and validation and inflates accuracy.

3. **Do not optimise generic HAR labels before product truth.**
   RuView needs truthful `absent`, `desk_seated`, `away`, and `uncertain` more than `run`, `pickup`, or `fall`.

4. **Do not let deep model confidence bypass the product guardrails.**
   The desk state should remain conservative; model confidence is evidence, not final truth.

5. **Do not collect more random CSI when the failure is decision-layer semantics.**
   If empty-office signatures repeat and are stable, label them as hard negatives and fix the decision layer / temporal model.

---

## Priority references to inspect next

1. **SenseFi benchmark**
   - <https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark>
   - likely first codebase to inspect for reusable preprocessing/model/eval patterns

2. **Time-Selective RNN for Device-Free Multiroom Human Presence Detection Using WiFi CSI**
   - listed under occupancy detection
   - likely relevant to temporal presence and room-level false positives

3. **SHARP / SHARPax**
   - <https://github.com/francescamen/SHARP>
   - <https://github.com/francescamen/SHARPax>
   - relevant to environment/person independence and domain shift discipline

4. **MM-Fi dataset**
   - <https://github.com/ybhbingo/MMFi_dataset>
   - relevant to multi-modal CSI+vision/radar supervision patterns, not direct deployment

5. **DeepSeg**
   - <https://github.com/ChunjingXiao/DeepSeg>
   - relevant to activity segmentation rather than whole-window classification

6. **EfficientFi**
   - <https://github.com/Marsrocky/EfficientFi>
   - relevant if RuView needs lightweight/compressed CSI features for edge/runtime

7. **ESP32 CSI Tool**
   - <https://stevenmhernandez.github.io/ESP32-CSI-Tool/>
   - useful comparison point for ESP32 collection assumptions

---

## Reference 3 — CSI-Bench: large-scale in-the-wild multitask WiFi sensing

Source: <https://ai-iot-sensing.github.io/projects/project.html>  
Code: <https://github.com/guozhen-jenn-zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark>  
Dataset: Kaggle, linked from the project page / repo  
Paper: arXiv `2505.21866`, listed by the repo as NeurIPS 2025 / NeurIPS volume 38.

### What it contains

CSI-Bench is the strongest external resource in this triage batch because it is not just a demo repo or resource list. It is a recent benchmark and dataset aimed at real-world CSI sensing:

- 460+ hours of CSI recordings
- 35 users
- 26 environments
- 16 device types
- homes, offices, and public indoor areas
- commercial WiFi edge devices / routers rather than only lab NIC setups
- tasks:
  - fall detection
  - breathing monitoring
  - room-level localization
  - motion source recognition
  - human activity recognition
  - user identification
  - proximity recognition
- benchmark code with local and SageMaker runners
- seven baseline model families:
  - MLP
  - LSTM
  - ResNet-18
  - Transformer
  - ViT
  - PatchTST
  - TimeSformer-1D
- dataset split files including in-distribution, difficulty splits, cross-device, cross-environment, and cross-user splits
- three-seed reproduction statistics in `RESULTS.md`

The codebase recently notes updated dataset/code to address data issues and says benchmark experiments were re-run on Kaggle Version 12. That is a positive reproducibility signal, but all reported numbers should still be treated as external claims until locally reproduced.

### Why it matters more than the earlier links

This is the first resource in the current batch that directly attacks RuView’s real gap:

> model behaviour under real homes, real users, different devices, different environments, and multitask labels.

The previous links support the idea of temporal windows and point to the literature. CSI-Bench gives a concrete benchmark shape RuView can copy:

- explicit dataset layout
- metadata per sample
- label mapping files
- split manifests
- OOD test splits
- multi-seed runs
- result aggregation
- confusion matrices / reports

For RuView, the useful artifact is not just a model. It is the **experiment harness discipline**.

### Useful ideas for RuView

#### 1. Treat split manifests as first-class artifacts

CSI-Bench stores splits under each task, including cross-device / cross-env / cross-user tests.

RuView should do the same for desk occupancy and office sensing:

- `train_id.json`
- `val_id.json`
- `test_id.json`
- `test_cross_day.json`
- `test_cross_node_layout.json` if node placement changes
- `test_cross_activity_context.json` for seated vs standing-near vs away
- `test_hard_empty_office.json` for known false-positive empty-room signatures

This is more valuable than adding another model before the evaluation shape is clean.

#### 2. Add motion-source recognition as a RuView hard-negative concept

CSI-Bench has a Motion Source Recognition task with human / pet / iRobot / fan-style categories.

RuView’s recurring false-positive problem is similar in spirit: stable RF motion may be non-human or room/background motion but still look person-like to a broad classifier.

RuView should promote this into its own calibration/evaluation idea:

- `human_motion`
- `chair_or_desk_motion`
- `fan_or_hvac_motion`
- `outside_room_motion`
- `empty_office_stable_rf_motion`
- `unknown_nonhuman_motion`

This could feed the conservative decision layer: non-human / broad uncertain motion should not create person instances or vitals.

#### 3. Multitask learning is probably relevant

CSI-Bench reports multitask gains for shared-data Human Activity Recognition, User Identification, and Proximity Recognition. The specific accuracy numbers are external claims, but the pattern is relevant.

RuView’s product state is also multi-task:

- presence
- desk seated
- near desk but not seated
- moving / broad activity
- proximity to desk/AP/node geometry
- vitals valid/null
- hard-negative motion source

A small shared temporal encoder with separate conservative heads may be better than separate brittle classifiers:

- `presence_head`
- `desk_state_head`
- `motion_source_head`
- `vitals_validity_head`
- `proximity_or_zone_head`

Important: user identification should remain out of scope unless explicitly opted in. CSI-Bench includes UID, but RuView should treat identity as a privacy risk / leakage audit signal, not a default product target.

#### 4. Model zoo choice is useful, but secondary

The model list is useful as a controlled comparison set:

- LSTM: temporal baseline
- ResNet-18 / 1D CNN: local temporal-frequency pattern baseline
- Transformer / ViT / PatchTST / TimeSformer-1D: attention/time-series baselines

For RuView, do not jump straight to the biggest model. Start with:

1. temporal feature baselines
2. LSTM or compact 1D CNN
3. PatchTST / Transformer only if the leak-safe splits show real headroom

#### 5. Their `win_len=500`, `feature_size=232` is a useful shape clue

The repo configs use `win_len=500` and `feature_size=232`. This confirms the broader pattern from the earlier HAR repo: WiFi sensing benchmarks usually work on fixed-length CSI windows, not individual frames.

RuView needs its own shape because ESP32/node/fused data is different, but this provides a concrete comparison point for the window extractor.

#### 6. Real-world commercial-device diversity validates RuView’s calibration stance

CSI-Bench spans many environments and device types. That supports the idea that deployment accuracy is dominated by domain shift:

- device shift
- environment shift
- user shift
- activity context shift
- background-motion shift

RuView should not aim for “one static model trained once.” It should aim for:

- repeatable calibration
- per-room/per-layout adapters
- hard-negative refreshes
- conservative output semantics
- OOD split monitoring

### Caveats / risks

1. **License ambiguity must be checked before reuse.**
   The extracted GitHub page reports MIT in GitHub chrome, while the README says CC BY-NC-ND 4.0. Treat the dataset/code as research reference only until the license files are inspected directly.

2. **Dataset schema still will not match RuView directly.**
   CSI-Bench uses commercial router/edge-device CSI and H5/MAT layouts. RuView uses ESP32 multistatic JSONL, node geometry, and product-specific desk labels.

3. **Reported accuracies are not RuView evidence.**
   Use their benchmark structure and model candidates. Do not cite their numbers as proof that RuView will work.

4. **UID is a privacy warning, not a product target.**
   Human identification success in CSI benchmarks reinforces that RF sensing can leak identity. RuView should keep identity opt-in and strengthen privacy/leakage auditing.

### RuView action from this reference

Update the proposed temporal-occupancy spike to copy CSI-Bench’s benchmark discipline:

- create a dataset manifest, not just ad-hoc files
- persist split manifests
- include OOD splits from day/layout/context
- run at least 3 seeds once a model exists
- emit confusion matrices and per-class reports
- include a hard-negative empty-office / non-human-motion test split
- keep model outputs advisory under the conservative decision layer

---

## Updated concrete next decision

Recommended next RuView research task:

> Build `benchmarks/desk-occupancy-temporal/README.md` plus a first extractor script that turns existing paired RuView JSONL + labels into per-node, label-pure windows and emits CSI-Bench-style manifest/split files.

Acceptance criteria for that first task:

- no model required yet
- proves we can form clean windows
- prints class counts per recording
- rejects mixed-label windows
- creates train/validation/test splits by recording/session
- includes one shape-check report showing window tensor dimensions per node

Only after that should we train a model. The harness is the leverage; the model choice is secondary.
