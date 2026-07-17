# CWT Implementation Plan for RL Drug-Cycling Analysis

This document describes a practical implementation plan for analyzing short-term and long-term drug-cycling dynamics from reinforcement-learning-driven fitness trajectories using a Continuous Wavelet Transform (CWT) pipeline.

The main goal is to separate slow baseline drift from time-localized oscillatory structure, then quantify whether cycling is dominated by short periods, long periods, or a mixture that changes over time. A CWT-first approach is appropriate because wavelets are well suited to nonstationary time series, and CWT provides a dense time-scale representation that is easier to interpret than a standard DWT when the objective is time-localized cycle analysis rather than compression or denoising alone.[cite:3][cite:18]

## Problem framing

The target signal is a trajectory such as normalized fitness over RL steps or Wright-Fisher generations, optionally paired with drug identity or switch events. The analysis should treat the trajectory as a combination of:

- a slow-moving baseline or trend,
- one or more oscillatory components,
- and stochastic noise.

A useful working model is:

\[
y(t) = m(t) + s(t) + \epsilon(t)
\]

where:

- \(y(t)\) is the observed fitness trajectory,
- \(m(t)\) is the slow baseline trend,
- \(s(t)\) is the oscillatory component to be analyzed with CWT,
- \(\epsilon(t)\) is residual noise.

This framing is consistent with prior work separating long-term fitness behavior from oscillatory dynamics induced by drug switching and adaptive control.[cite:2]

## Recommended implementation strategy

The coding agent should build the analysis in stages so each layer can be validated independently. The recommended sequence is:

1. Standardize and validate the input time series.
2. Ensure uniform sampling in time.
3. Apply optional light denoising if the trajectories are very noisy.
4. Estimate and remove a slow baseline trend.
5. Compute the CWT of the detrended residual.
6. Convert wavelet scales into interpretable periods or frequencies.
7. Aggregate wavelet power into short-cycle and long-cycle bands.
8. Extract summary metrics and event-aligned statistics.
9. Export plots and machine-readable summary files.

This order keeps preprocessing, decomposition, and interpretation separate, which makes debugging much easier.

## Input data contract

The pipeline should accept either a Pandas DataFrame or CSV files with the following minimum schema:

| Column | Required | Description |
|---|---|---|
| `t` | Yes | Time index, generation, or RL step |
| `fitness` | Yes | Observed scalar trajectory value |
| `run_id` | Yes | Unique trajectory identifier |
| `policy_id` | No | Policy label for grouping and comparison |
| `drug_action` | No | Drug identity at each time step |
| `switch_flag` | No | Binary flag marking action switches |
| `state_summary` | No | Optional compressed state descriptor |

The code should sort data by `run_id` and `t`, check for duplicates, and verify monotonic time. If there are missing time points, the code should either resample or explicitly fail with a validation error instead of silently proceeding.

## Suggested code structure

A modular layout will let another agent swap methods without rewriting the entire analysis stack.

```text
analysis/
  io.py
  preprocess.py
  wavelets.py
  features.py
  events.py
  plots.py
  report.py
  config.py
  tests/
    test_synthetic_signals.py
    test_preprocessing.py
    test_scale_period_conversion.py
```

Suggested responsibilities:

- `io.py`: file loading, schema validation, sorting, resampling.
- `preprocess.py`: smoothing, normalization, trend estimation, detrending.
- `wavelets.py`: CWT wrapper, scale generation, scale-to-period conversion, cone-of-influence logic.
- `features.py`: bandpower, peak detection, ridge extraction, entropy or concentration measures.
- `events.py`: event-locked analysis around drug switches.
- `plots.py`: trajectory, residual, scalogram, bandpower, and summary visualizations.
- `report.py`: assemble run-level and policy-level summary tables.
- `config.py`: centralized parameters and defaults.

## Preprocessing plan

### 1. Uniform sampling

Wavelet methods assume regular sampling in time, so trajectories should be resampled onto a constant `dt` grid before any spectral analysis.[cite:25]

Implementation notes:

- If the data are already on integer time steps, keep them as-is.
- If there are gaps, use interpolation only if the gaps are small and documented.
- If sampling is irregular in a severe way, reject the series and request preprocessing upstream.

### 2. Optional denoising

Only use light denoising. The goal is not to remove oscillations, but to reduce impulse noise or simulation artifacts.

Recommended options:

- rolling median for spike suppression,
- short Savitzky-Golay filter for smooth noise reduction,
- or no denoising if the raw trajectories are reasonably stable.

The denoiser should be optional and parameterized.

### 3. Trend estimation

The baseline trend should be estimated before the CWT so that low-frequency drift does not masquerade as long-period cycling.

Recommended baseline methods:

- rolling mean,
- LOESS/LOWESS,
- spline smoothing.

The simplest starting point is a centered rolling mean with configurable window size. A stronger second option is LOWESS when the baseline shape is nonlinear.

### 4. Detrending

Compute:

```python
residual = fitness - trend
```

Optionally store a normalized residual such as z-scored residual for cross-run comparisons.

## Wavelet stage

### 1. Wavelet choice

Use a complex Morlet wavelet as the first implementation choice. Morlet wavelets are commonly used for time-frequency analysis of nonstationary oscillatory signals and give a good balance between time and frequency resolution.[cite:18][cite:21]

### 2. Scale grid

Use logarithmically spaced scales rather than linear spacing so the analysis has reasonable resolution across short and long timescales.[cite:25]

Recommended initial configuration:

- `num_scales = 64` or `96`,
- `voices_per_octave = 12` if supported,
- `period_min = 2 * dt`,
- `period_max = min(T / 3, configured_limit)`.

### 3. Scale-to-period conversion

The code should always convert raw scales into periods or frequencies immediately after the transform. All downstream summaries should be done in period space because the scientific question is about cycle lengths rather than abstract scale indices.[cite:25]

### 4. Cone of influence

The code should compute and store the cone of influence and either mask or downweight coefficients near the edges. Edge effects are a standard issue in wavelet analysis, and unmasked edge power is easy to over-interpret.[cite:25]

## Band definitions

Cycle bands should be defined in terms of period ranges and linked to the drug update interval when that quantity is known.

A good starting scheme is:

- **Short-term band:** periods from `2 * update_interval` to `10 * update_interval`
- **Medium-term band:** periods from `10 * update_interval` to `50 * update_interval`
- **Long-term band:** periods above `50 * update_interval`

If the update interval is not explicit, define bands relative to the observed trajectory length and revise after inspecting the first batch of scalograms.

If the policy alternates drugs at a fixed cadence, at least one band should explicitly cover the nominal switch period and potentially its harmonics.[cite:2]

## Features to extract

The agent should compute both time-local and trajectory-level summaries.

### Time-local features

For each time point:

- dominant period,
- dominant power,
- short-band power,
- medium-band power,
- long-band power,
- short/long dominance label.

### Run-level features

For each run:

- global peak period from time-averaged power,
- mean short-band power,
- mean long-band power,
- short-to-long bandpower ratio,
- fraction of time each band dominates,
- standard deviation of dominant period over time,
- spectral entropy or concentration score,
- number and duration of persistent ridges.

### Policy-level features

Aggregate across runs within each `policy_id`:

- mean and confidence interval for peak period,
- mean and confidence interval for short/long bandpower,
- distribution of dominant band labels,
- between-run variance.

## Event-aligned analysis

If `drug_action` or `switch_flag` is available, the pipeline should include event-locked summaries around switch times. This is important because prior work found that therapeutic update timing strongly influences long-term behavior.[cite:2]

Recommended event analysis:

- extract windows around each switch event,
- average detrended residual across events,
- average short-band and long-band power around events,
- compare pre-switch versus post-switch power,
- stratify by switch type if multiple drug pairs exist.

This allows the analysis to distinguish policy-driven oscillations from slower adaptive dynamics.

## Outputs

The implementation should save both plots and structured data files.

### Per-run outputs

- raw trajectory with trend overlay,
- detrended residual plot,
- CWT scalogram with cone-of-influence marking,
- bandpower-over-time plot,
- optional event-aligned plot.

### Aggregated outputs

- summary CSV with one row per run,
- policy summary CSV,
- representative or average scalogram per policy,
- peak-period histogram,
- short-versus-long bandpower comparison plot.

A three-panel plot is especially useful:

1. raw fitness and trend,
2. detrended residual,
3. scalogram with annotated short and long bands.

## Pseudocode

```python
# Load and validate
.df = load_trajectories(path_or_df)
.df = validate_schema(df)
.df = sort_and_resample(df, dt=cfg.dt)

run_metrics = []

for run_id, run in df.groupby("run_id"):
    t = run["t"].to_numpy()
    y = run["fitness"].to_numpy()

    y_smooth = maybe_denoise(y, method=cfg.denoise_method, params=cfg.denoise_params)
    trend = estimate_trend(y_smooth, method=cfg.trend_method, params=cfg.trend_params)
    resid = y - trend

    scales = make_log_scales(
        dt=cfg.dt,
        period_min=cfg.period_min,
        period_max=cfg.period_max,
        num_scales=cfg.num_scales,
    )

    coeffs, power, periods, coi = compute_cwt(
        resid,
        scales=scales,
        wavelet=cfg.wavelet,
        dt=cfg.dt,
    )

    valid_mask = build_coi_mask(t, periods, coi)
    power_masked = apply_mask(power, valid_mask)

    short_power = compute_bandpower(power_masked, periods, cfg.short_band)
    medium_power = compute_bandpower(power_masked, periods, cfg.medium_band)
    long_power = compute_bandpower(power_masked, periods, cfg.long_band)

    dominant_period_t = periods[np.nanargmax(power_masked, axis=0)]
    global_peak_period = estimate_global_peak_period(power_masked, periods)
    ridge_stats = extract_ridges(power_masked, periods)

    metrics = {
        "run_id": run_id,
        "global_peak_period": global_peak_period,
        "short_power_mean": np.nanmean(short_power),
        "medium_power_mean": np.nanmean(medium_power),
        "long_power_mean": np.nanmean(long_power),
        "short_long_ratio": np.nanmean(short_power) / (np.nanmean(long_power) + 1e-9),
        "dominant_period_mean": np.nanmean(dominant_period_t),
        "dominant_period_std": np.nanstd(dominant_period_t),
        "spectral_entropy": compute_spectral_entropy(power_masked),
        **ridge_stats,
    }

    if "switch_flag" in run.columns:
        event_stats = compute_event_locked_metrics(
            residual=resid,
            short_power=short_power,
            long_power=long_power,
            switch_flag=run["switch_flag"].to_numpy(),
            window=cfg.event_window,
        )
        metrics.update(event_stats)

    save_run_plots(run_id, t, y, trend, resid, power_masked, periods, coi)
    run_metrics.append(metrics)

summary_df = pd.DataFrame(run_metrics)
save_summary(summary_df)
save_policy_aggregates(summary_df)
```

## Configuration template

The coding agent should centralize all major settings in one config object or YAML file.

```yaml
dt: 1
wavelet: cmor
num_scales: 96
period_min: 2
period_max: null
trend_method: loess
trend_params:
  frac: 0.05
denoise_method: none
denoise_params: {}
short_band: [2, 10]
medium_band: [10, 50]
long_band: [50, 200]
event_window: 50
normalize_residual: false
mask_coi: true
```

If bands depend on update interval, compute them programmatically from metadata instead of hard-coding them.

## Recommended Python stack

Suggested libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn` for summary figures if desired
- `statsmodels` for LOWESS if used
- `PyWavelets` or `ssqueezepy` for CWT

The exact wavelet package is less important than consistent scale-period conversion, masking of edge regions, and stable plotting conventions.

## Validation plan

The coding agent should validate the pipeline on synthetic signals before running it on the RL data. Suggested synthetic test cases:

1. pure sinusoid with known period,
2. sum of two sinusoids with different periods,
3. chirp with gradually changing period,
4. slow trend plus sinusoid,
5. fixed switching-period signal plus noise,
6. white noise only.

Expected validation behavior:

- the known period should appear as the dominant period,
- a chirp should show a moving ridge,
- trend-only components should be suppressed after detrending,
- white noise should not create stable narrow ridges.

This validation step is necessary because CWT results are visually intuitive but can still be misread if detrending, normalization, or edge handling is incorrect.[cite:3][cite:25]

## Interpretation rules for the final analysis

The agent should encode the following guardrails into the analysis notebook or README:

- Do not interpret power inside the cone of influence as strong evidence.
- Do not interpret low-frequency power in the raw signal as cycling until detrending has been checked.
- Require persistence over time before calling a pattern a meaningful cycle.
- Compare observed peak periods to the known drug update interval before attributing the pattern to biology or learning dynamics.
- Treat broad diffuse low-frequency power as possible drift rather than clean periodicity unless a stable ridge is present.

## Deliverables checklist for the coding agent

The final implementation should provide:

- a reusable preprocessing module,
- a CWT module with scale-to-period conversion,
- bandpower and peak-extraction features,
- optional event-locked switch analysis,
- run-level CSV summaries,
- policy-level CSV summaries,
- saved diagnostic plots,
- synthetic validation tests,
- and one end-to-end script or notebook showing how to run the full pipeline.

## Handoff summary

The recommended first implementation is a Morlet-based CWT pipeline on detrended fitness trajectories, with power summarized in short and long period bands, plus event-locked analysis around drug switches when available.[cite:18][cite:25][cite:2] This design is well suited to RL-controlled drug cycling because it preserves time-localized information and can separate transient switch-driven oscillations from slower adaptation of the evolving population.[cite:3][cite:17][cite:2]
