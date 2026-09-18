# flypair — PLAN

Written after cloning and reading the four reference repos (in `reference/`,
git-ignored) and the two small annotation tables. Everything below marked
**verified** was read from those sources on 2026-09-17; nothing is guessed.

## 1. Data sources found (verified)

### Model constants — `reference/Drosophila_brain_model/model.py` (Shiu et al. 2024)
| constant | value | source line |
|---|---|---|
| v_0 (rest) | -52 mV | `default_params['v_0']` |
| v_rst (reset) | -52 mV | `default_params['v_rst']` |
| v_th | -45 mV, strict `v > v_th` | `'v_th'`, `'eq_th'` |
| t_mbr (tau_m) | 20 ms | `'t_mbr'` |
| tau (tau_syn) | 5 ms, alpha-type `dg/dt=-g/tau`, `on_pre: g += w` | `'tau'`, eqs |
| t_rfc | 2.2 ms (`unless refractory` on both v and g) | `'t_rfc'` |
| t_dly | 1.8 ms synaptic delay | `'t_dly'` |
| w_syn | 0.275 mV per synapse (× signed synapse count) | `'w_syn'` |
| Poisson input | `r_poi`=150 Hz, weight `w_syn*f_poi` = 68.75 mV **onto v directly**; driven neurons get `rfc = 0` | `poi()` |
| silencing | zero all synapses **from** the neuron (`syn.w['i==i']=0`) | `silence()` |
| dt | not set in model.py → Brian2 default **0.1 ms** (confirmed by drosophila-brain-mlx `core.DT = 0.1`, tick-by-tick parity with Brian2) | |
| integration | Brian2 `method='linear'` exact update; coefficients transcribed in mlx `core.py` | |
| tick order | refractory decrement → exact v/g update (non-refractory only) → threshold → read 18-tick delay ring → scatter signed counts × w_syn onto g (dropped if postsynaptic refractory) → Poisson on v → reset/rfc reload/ring store | mlx `engine_naive.tick` |

Reference result for the sanity check (from the shipped `results/example/*.parquet`,
30 trials × 1 s, 21 right sugar GRNs; IDs listed in `example.ipynb`):
- 150 Hz drive → MN9 (`720575940660219265`) **93.3 Hz**; 100 Hz drive → **67.0 Hz**.

### FlyWire (female, brain only) — Shiu repo ships v783 files
- `Completeness_783.csv`: index = root_id (138,639 rows), column `Completed`.
- `Connectivity_783.parquet` (15.09 M rows): `Presynaptic_ID, Postsynaptic_ID,
  Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory,
  Excitatory x Connectivity` (already signed). Index = row of Completeness csv.
- Annotations (NOT in Shiu repo): Schlegel et al. 2024,
  `https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv`
  columns used: `root_id, super_class, cell_class, cell_type, hemibrain_type,
  side (left/right/center/na), top_nt, synonyms`. 138,625 of 138,639 model
  neurons have a row.
- Shiu's 21 sugar GRNs = `cell_type == LB3`, `side == left` in the TSV (Shiu
  calls them "right"; the TSV's side convention is mirrored — documented).
- MN9 = `cell_type == CB0701` (2 neurons; verified via Shiu's MN9 ID).
- vpoDN = `hemibrain_type == vpoDN` (cell_type DNp37, 2 neurons).

### MaleCNS v1.0 (male, brain + VNC) — URLs from `fly.ai/flybrain/build.py` and mlx
- bucket `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14.5 MB)
  - `body-neurotransmitters-male-cns-v1.0.feather` (43 MB, cols `body, consensus_nt`)
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.05 GB, cols `body_pre, body_post, weight`)
- Node selection: `superclass` non-null → **166,700 neurons** (fly.ai & mlx agree).
- Annotation columns used: `bodyId, type, flywireType, hemibrainType, class,
  superclass, instance, somaSide, rootSide, synonyms, receptorType, dimorphism`.
  Side = somaSide → rootSide → `_L/_R` suffix of instance (sensory neurons have
  no soma side; fly.ai does the same fallback).
- `consensus_nt` labels among the 166,700: acetylcholine 103,720; glutamate
  29,302; gaba 22,069; histamine 7,891; unclear 2,999; dopamine 392; NaN 178;
  octopamine 101; serotonin 48.
- Sign convention (fly.ai): label contains `gaba|glutamate|histamine` → −1,
  everything else +1. Option `unknown_nt: drop` gives the mlx convention
  (acetylcholine +1, gaba/glutamate −1, all else dropped).
- Sugar GRNs on MaleCNS: types `LB3b`, `LB3c` (Tastekin et al. 2025 ↔ Gr64f;
  used by mlx `activity_film.py`, instances `LB3b_R`, `LB3c_R` = 17 neurons).
  mlx reports MN9_L ≈ 53 Hz at 100 Hz drive (their seed).

### Group name availability (probed against both annotation tables)
| group | malecns | flywire |
|---|---|---|
| sugar GRN | `^LB3[bc]$` | `^LB3$` (side L) |
| bitter GRN | **unresolved** (LB1/LB2/LB4 exist but no receptor label; not claimed) | unresolved |
| Gr32a | **unresolved** (no such label anywhere) | unresolved |
| ppk23 | `receptorType == putative_ppk23` (269, VNC leg GRNs) | unresolved |
| Or67d/cVA ORN | `^ORN_DA1$` | `^ORN_DA1$` |
| JO-A/B | `^JO-[AB]` | `^JO-[AB]` |
| LC10a | `^LC10a$` | `^LC10a$` |
| LPLC2 | `^LPLC2$` | `^LPLC2$` |
| visual generic | lamina `^L[12]$` by side | same |
| P1/pC1 | `^pC1` with `dimorphism` ~ male-specific | `^pC1[a-e]$` (female pC1) |
| mAL | `^mAL` | `^mAL` |
| vAB3 | synonyms ~ `vAB3` (types AN09B017e/f/g) | unresolved |
| aSP | `^aSP` | `^aSP` |
| vpoEN | `^vpoEN$` | `^vpoEN$` |
| pIP10 | `^pIP10$` | unresolved (not in female) |
| wing MNs | `^(b[123]|i[12]|iii[13]|hg[1-4]|ps[12]|tp[12]|tpn|hi[12]|hiii2) MN$|^DLMn|^DVMn|^hDVM MN$` | n/a (no VNC) |
| DNp01 | `^DNp01$` | `^DNp01$` |
| DNa01/DNa02 | yes | yes |
| forward DNs | `^(DNp09|DNg100)$` | same |
| MDN | `^MDN$` | `^MDN$` |
| MN9 | `^MN9$` | `^CB0701$` |
| aggression DNs | **unresolved** (no aggression annotation; aIPg exist as internal) | pC1d/e, aIPg exist (internal only) |
| vpoDN | unresolved (male) | `hemibrain_type ^vpoDN$` |
| DNp13 | `^DNp13$` | `^DNp13$` |

## 2. Architecture

```
flypair/
  constants.py     Shiu constants + exact linear-update coefficients
  connectome/      base.Connectome (signed CSR float32 post×pre, neuron table, cache npz+parquet)
                   tiny.py  malecns.py  flywire.py  shuffle.py (degree-preserving, from mlx idea)
  groups.py        registry: groups/<connectome>.yaml -> indices; report; fails on 0 matches
  brain.py         torch LIF, batch [n_flies, n_neurons], CSR spmm or event-driven gather (benchmarked)
  inputs.py        InputSource interface (sensory drive + emitters); default = world/channels
  world.py         arena, poses, emitters, motor decoders (all hand-designed, listed in README)
  channels.py      declarative source->target mappings (gain, falloff, angular gating, delay, clip)
  scenario.py      YAML schema validation + run loop; multi-connectome (one Brain per connectome)
  controls.py      playback / open_loop / shuffled; metrics.py coupling metric
  record.py plots.py video.py dub.py cli.py
groups/*.yaml  scenarios/*.yaml  tests/  flypair_colab.ipynb
```

Brain step (per dt = 0.1 ms), all tensors [B, N]:
1. rfc -= 1; not_ref = rfc == 0
2. v,g exact update where not_ref
3. spike = not_ref & v > v_th ; spike &= ~silence_mask (per fly)
4. delayed = ring[t % 18]; I = propagate(delayed) (CSR spmm or gather) × w_syn × fly_gain
5. g += I where not_ref ; v += w_ext × poisson_draw where not_ref (driven idx only)
6. reset, rfc reload (0 for driven neurons, as Shiu), ring store, accumulate counts

World tick every 20 ms (200 brain steps): group spike counts → Hz → motor
decoders → poses/emitters → channels → target Poisson rates for the next window.

## 3. Open risks
- Free-tier Colab RAM (~12 GB): MaleCNS build streams the 1 GB feather in
  batches (as fly.ai). Expected peak < 4 GB. CSR + transpose ≈ 400 MB.
- Speed: 10,000 brain steps per biological second. T4 spmm on 25 M nnz × B
  ≈ 1–3 ms/step → ~20 s per bio-second. CPU-only Colab could be 10× slower;
  the notebook benchmarks and picks. Scenarios default to short durations.
- Shiu constants were fit to FlyWire; on MaleCNS (with VNC) they are reused
  unchanged (as mlx notes). Motor readouts on MaleCNS may saturate.
- Motor decoders and sensory encoders are hand-designed. Playback/shuffled
  controls are the only guard against over-interpretation.
- Gr32a, bitter GRNs, aggression DNs: not annotated → channels that need them
  are skipped with a loud warning (or `on_unresolved: fail`).
- FlyWire side labels are mirrored relative to Shiu's naming.
- M7 (interactive steering): deferred; only the `InputSource` seam is built now.
