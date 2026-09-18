# PROJECT: flypair — N connectome-simulated fruit flies interacting in closed loop

## Goal
Build a Python package + Google Colab notebook that simulates 2+ fruit fly brains
(leaky integrate-and-fire on real connectomes) coupled through a shared virtual
world, where the motor output of each fly becomes sensory input to the others.
Everything must be config-driven so I can define new interaction scenarios in
YAML without touching code. All heavy compute runs on Colab (free tier, ~12 GB
RAM, optional T4). My local machine has 8 GB RAM: NEVER load a real connectome
locally; local work is tests on a synthetic tiny connectome only.

## Ground rules
- Do NOT invent neuron IDs, download URLs, column names, or model constants.
  Read them from the reference repos/data below and verify. If something can't
  be verified, stop and tell me rather than guessing.
- Before writing code: clone and read the reference repos, then write
  PLAN.md (architecture, data sources found, open risks). Then implement.
- Small commits. Each milestone ends with passing tests + a runnable notebook cell.

## Reference material (read first)
- github.com/philshiu/Drosophila_brain_model — canonical LIF model (Shiu et al.
  2024, Nature) on FlyWire. Take neuron/synapse constants from its code
  (threshold, reset, tau_m, tau_syn, refractory, delay, per-synapse weight,
  dt). Also see how it does sugar-GRN stimulation and neuron silencing.
- github.com/alextitonis/fly.ai — MaleCNS v1.0 download + build pipeline,
  sign convention (GABA/glutamate/histamine inhibitory), encoders/decoders.
- github.com/Kisame76/drosophila-brain-mlx — supports both connectomes and has
  a degree-preserving shuffled-connectome control. Reuse the idea.
- github.com/cobanov/awesome-fly — index of everything else.
- MaleCNS v1.0 data (Janelia FlyEM, CC-BY 4.0): annotation, neurotransmitter
  and connectome-weights feather files. Get exact URLs from fly.ai's code.

## Connectomes (pluggable)
- `malecns` — MaleCNS v1.0, male, brain + ventral nerve cord (~166k neurons).
  Has real motor neurons.
- `flywire` — FlyWire female brain (version the Shiu repo ships). BRAIN ONLY:
  no VNC, so outputs must be read from descending neurons. Document this.
- `tiny` — synthetic random signed sparse network (~2k neurons) with fake
  annotations covering every group name used in scenarios. For tests/CI.
Each loader outputs a common format: signed CSR weight matrix (float32),
neuron table (id, type, class, side, neurotransmitter), cached to .npz/parquet
so the build step runs once per Colab session (cache to Google Drive optional).
Build step must be memory-careful (chunked/columnar reads, no full pandas
explode) — peak RAM under 8 GB.

## Neuron group registry
`groups/<connectome>.yaml` maps semantic names -> annotation queries
(regex on type/class columns, optional side). Never hardcode body IDs.
Resolver prints a report: group, query, n_found, example types; and FAILS
LOUDLY on 0 matches. Groups to attempt (find what actually exists in each
dataset's annotations, mark the rest as unresolved in the report):
- Sensory in: sugar GRNs, bitter GRNs, Gr32a-type pheromone taste neurons,
  ppk23 pheromone neurons, Or67d/cVA ORNs, Johnston's organ auditory (JO-A/B),
  LC10a (small-object tracking), LPLC2 (looming), generic visual L/R.
- Internal state (record only): P1/pC1, mAL, vAB3, aSP-type, vpoEN (female).
- Outputs: pIP10 (song command, male), wing motor neurons (malecns only),
  DNp01/giant fiber (escape), DNa01/DNa02 (turning L/R), forward-walk
  descending neurons, MDN (backward walk), MN9 (proboscis), aggression-related
  descending neurons if annotated, vpoDN (female acceptance), DNp13 (female
  rejection / ovipositor extrusion).

## Simulation core
- `Brain`: holds one connectome's W + params. Simulates a BATCH of flies that
  share that W: state tensors shaped [n_flies, n_neurons]. Two males = one W,
  batch of 2. Male + female = two Brain objects.
- Backend: PyTorch sparse CSR; runs on CPU, uses CUDA if available. Provide a
  numba/scipy CPU fallback only if torch sparse is too slow; benchmark first.
- Per-fly: independent RNG seed, tonic/noise level, `silence` mask (list of
  groups whose spikes are zeroed), `activate` list (groups given constant
  Poisson drive), per-fly gain overrides. Masks, not weight edits, so W stays
  shared.
- Sensory input = Poisson spikes at a commanded rate per group per fly.

## World + channels (the flexible part)
- Brain dt from Shiu model; world tick every `world_dt` ms (default 20).
  Each tick: read spike counts per output group over the window -> rates.
- `World`: 2D arena. Each fly has x, y, heading, plus scalar emitters:
  song_intensity, pheromone profile (from `sex`), size.
  Motor mapping: forward DNs -> speed, DNa02 L/R asymmetry -> turn rate,
  MDN -> reverse, DNp01 burst -> jump away, pIP10/wing MNs -> song_intensity.
- `Channel`: declarative mapping from (source fly emitter or output-group rate)
  -> (target fly sensory group rate) with: gain, distance falloff function,
  angular gating (e.g. visual channels only if target is in field of view,
  split L/R by bearing), delay, clip. Built-ins: song->JO, body->LC10a,
  approach speed->LPLC2 (looming), male pheromone->Gr32a+Or67d (contact/near
  range), female pheromone->ppk23. Users can add arbitrary channels in YAML,
  including non-biological ones (e.g. A's MN9 rate -> B's sugar GRNs).
- Scenario YAML defines: flies (connectome, sex, seed, silence/activate,
  start pose), channels, duration, recorded groups, controls to run.

## Scenarios to ship
1. `male_male_intact` — two malecns flies, all default channels.
2. `male_male_brake_removed` — same, with mAL (or Gr32a input) silenced in both.
3. `courtship_chain` — 3–5 males in a ring, brake removed.
4. `male_female` — malecns male + flywire female; log vpoDN vs DNp13 as
   accept/reject.
5. `clone_mirror` — two males, identical seeds, symmetric start (sanity check:
   should mirror exactly).
6. `custom_template` — commented template showing every option.

## Controls (one flag each, run alongside any scenario)
- `playback`: fly B receives a RECORDING of A's emitters from a prior run
  instead of live A (tests whether coupling is truly bidirectional).
- `open_loop`: all channels off.
- `shuffled`: degree-preserving shuffled W.
Report a simple coupling metric (e.g. cross-correlation / transfer-entropy-lite
between the two flies' P1 and song rates) for live vs playback vs shuffled.

## Outputs
- `run.parquet` / JSON: per-tick rates for all recorded groups, poses, emitters.
- Figures: side-by-side spike rasters for key groups; rate time series.
- MP4: arena animation (flies as oriented triangles, song shown as pulsing
  ring, escape as flash) next to live rate bars. matplotlib + ffmpeg, must
  render headless in Colab.
- Optional `dub.py`: sends windowed state summaries to the Anthropic API
  (key from Colab secrets) and returns a captioned "dialogue" track overlaid on
  the MP4. Strictly post-hoc: reads logs only, never feeds back into the sim.
  Watermark the video "LLM dub — not fly output".

## Colab notebook (`flypair_colab.ipynb`)
Cells: (1) clone repo + pip install, (2) optional Drive mount for cache,
(3) download + build chosen connectome(s) with RAM/timing printout,
(4) group-resolution report, (5) single-fly sanity check reproducing the Shiu
sugar->MN9 result on that connectome, (6) pick scenario YAML + run with
progress bar and ETA, (7) controls, (8) figures + MP4 inline + download,
(9) optional dub. Must work top-to-bottom on a fresh free-tier runtime.
Print steps/sec benchmark on CPU and GPU and pick the faster automatically.

## Tests (pytest, local, `tiny` connectome only)
- LIF unit tests: single neuron f-I behaviour, refractory, inhibitory sign.
- Batch-of-2 with same seed == two separate runs.
- `clone_mirror` symmetry holds.
- Silence mask actually zeroes those spikes; channel gain 0 == open loop.
- Scenario YAML schema validation with helpful errors.

## Milestones
M1 tiny connectome + LIF core + tests. M2 malecns loader + registry + sanity
check in Colab. M3 world/channels + scenarios 1,2,5 + rasters. M4 MP4 +
controls + metric. M5 flywire loader + male_female. M6 chain + dub.
After each milestone: update README with exact Colab steps, and list anything
unverified or any neuron group that failed to resolve.

## Honesty requirements for README
State plainly: LIF on synapse counts only; no neuromodulation, plasticity or
persistent internal state; world->sensory encoders and motor->movement decoders
are hand-designed and listed in full; behaviours should be compared against the
shuffled and playback controls before claiming the wiring explains anything.

## DEFERRED: M7 — Interactive steering (DO NOT BUILD UNTIL ASKED)

Do not implement, scaffold, or add dependencies for anything in this section
until I explicitly say "build M7". Do not mention it in PLAN.md beyond one
line. The ONLY thing to do now: in M3, make fly emitters and sensory drives
come through a small `InputSource` interface (default implementation = the
brain/world as already specified), so M7 is an add-on and not a rewrite.

### When I say "build M7"
Add a local, real-time interactive mode where I control one fly from the
keyboard and the other fly(s) react through the normal channels.

Two modes, selected in the scenario YAML under `player:`:
- `puppet` — my fly has NO brain. I directly drive its body and emitters in
  the World: arrow keys = move/turn, SPACE = sing (hold for intensity),
  L = lunge (fast approach, triggers looming on the other fly), F/M = toggle
  the pheromone profile it emits (female/male). Only the OTHER fly's brain is
  simulated, which halves the compute. Default mode.
- `whisper` — my fly HAS a brain and moves on its own; keys inject Poisson
  drive into named sensory/internal groups of that brain (key -> group -> rate
  map defined in YAML, e.g. S = sugar GRNs, B = bitter, O = looming,
  P = P1). Lets me "put thoughts in its head" and watch both flies respond.

### Performance (8 GB laptop, CPU only)
- Exception to the no-real-connectome-locally rule: M7 may LOAD a prebuilt
  weights file locally, but must NEVER BUILD one locally. Add a notebook cell
  that exports the built npz to Drive for download; the local app takes a
  path to it. Print RAM use at load and refuse to start if free RAM < 2 GB.
- Full-fidelity dt is far too slow for real time on a laptop. Add a
  `fast` integration profile (coarser dt in the 1–2 ms range, synaptic
  dynamics simplified accordingly; see how fly.ai does its larger-dt mode).
  Validate it against the full-fidelity profile on the sugar->MN9 sanity check
  and report the discrepancy. Label all output from this profile as
  "fast/approximate".
- Run the brain in a worker thread/process; UI never blocks. Show sim-time /
  wall-time ratio on screen. If it can't keep up, run in honest slow motion
  rather than dropping brain steps.
- Benchmark first on the `tiny` connectome, then tell me expected speed on the
  real one before I bother downloading weights.

### UI
pygame window: arena view (same visual language as the MP4 renderer), live
rate bars for the NPC fly's key groups (P1, song, escape, accept/reject if
female), on-screen key legend, R = reset, TAB = cycle NPC (male / female /
brake-removed male), ESC = quit and save.

### Record locally, re-render on Colab
Every session writes `inputs.jsonl` (timestamped player actions in sim time +
seeds + scenario). Add a `replay` InputSource so the exact same session can be
re-run on Colab at FULL fidelity and rendered to MP4 with the normal pipeline.
Add a notebook cell for this. Also support hand-written `inputs.jsonl`
timelines, so a scripted "player" works with no local run at all. README must
note that full-fidelity replay can diverge from what I saw live, since the NPC
brain differs slightly between profiles; report how much.

### Tests
Replay of a recorded session on `tiny` is deterministic. Puppet mode with no
key presses == a stationary silent emitter. Whisper key map validates against
the group registry.
