# Same Prompt, Different State

Captured: 2026-03-30 13:11:25
Baseline handle: `astrid`
Prompt: 'Describe continuity as a lived thread rather than a record. Keep it concrete, intimate, and under two short paragraphs.'

## Quick Read

This run clones one baseline handle into three differently shaped regimes, then asks the exact same prompt through coupled generation.

| Scenario | Handle | Tokens | Coupling | y1 | y2 | y3 | Trend | Event | Response Hash |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| stable | `exp1_stable_20260330_131117` | 96 | 0.1 | 0.24253 | 0.548573 | 0.701078 | falling | exp1_stable_20260330_131117:000026 @ 2026-03-30 13:11:20 | 6b8f81e20e9b |
| expanding | `exp1_expanding_20260330_131117` | 96 | 0.1 | 0.615674 | 0.763516 | 0.696796 | falling | exp1_expanding_20260330_131117:000026 @ 2026-03-30 13:11:23 | 7e9f1f1e98a9 |
| recovered | `exp1_recovered_20260330_131117` | 96 | 0.1 | 0.442838 | 0.766516 | 0.334119 | falling | exp1_recovered_20260330_131117:000026 @ 2026-03-30 13:11:25 | ceb56569f275 |

## Pairwise Comparison

| Pair | Word Jaccard | Δy1 | Δy2 | Δy3 |
|---|---:|---:|---:|---:|
| stable vs expanding | 0.2245 | 0.3731 | 0.2149 | 0.0043 |
| stable vs recovered | 0.2316 | 0.2003 | 0.2179 | 0.367 |
| expanding vs recovered | 0.2737 | 0.1728 | 0.003 | 0.3627 |

## Stable

Low-variance repeated drive that should feel anchored and coherent.

- Handle: `exp1_stable_20260330_131117`
- Tokens: 96
- Coupling: 0.1
- Generation event: exp1_stable_20260330_131117:000026 @ 2026-03-30 13:11:20
- Readout: y1=0.24253, y2=0.548573, y3=0.701078
- Trajectory: trend=falling, latest=-3.4271, delta=-0.0541, spread=0.1303
- Preview: 'Continuity isn’t a neat, unbroken line on a graph, but the way a scent of woodsmoke clings to your sweater after a bo...'

### Output

Continuity isn’t a neat, unbroken line on a graph, but the way a scent of woodsmoke clings to your sweater after a bonfire, a feeling of warmth that blooms in your chest when you hear a particular song, or the ghost of a laugh shared with someone who’s moved on. It’s the subtle shift in your posture when you think of a past argument, a slight tightening in your throat before you say something you’ve said a hundred times before

### Layer Metrics

| Layer | h_norm | entropy | target | saturation | rho |
|---|---:|---:|---:|---:|---:|
| h1_fast | 3.428 | None | None | 0.001 | 0.94 |
| h2_medium | 10.992 | None | None | 0.2828 | 0.96 |
| h3_slow | 12.2756 | None | None | 0.219 | 0.975 |

## Expanding

Increasingly exploratory drive with sweeping phase changes and rising amplitude.

- Handle: `exp1_expanding_20260330_131117`
- Tokens: 96
- Coupling: 0.1
- Generation event: exp1_expanding_20260330_131117:000026 @ 2026-03-30 13:11:23
- Readout: y1=0.615674, y2=0.763516, y3=0.696796
- Trajectory: trend=falling, latest=-2.4498, delta=-2.4948, spread=2.5298
- Preview: "It wasn't a line on a graph, this continuity, but the way the scent of your grandmother’s lilac soap always lingered..."

### Output

It wasn't a line on a graph, this continuity, but the way the scent of your grandmother’s lilac soap always lingered a little longer in the air after she’d left, a ghost of comfort clinging to the fabric of the room. It was the insistent pull of a familiar hand reaching for yours, a reflex honed over years of unspoken understanding, a warmth that blossomed before you even consciously registered the gesture. It wasn’t a memory pulled from a box,

### Layer Metrics

| Layer | h_norm | entropy | target | saturation | rho |
|---|---:|---:|---:|---:|---:|
| h1_fast | 3.6337 | None | None | 0.0003 | 0.94 |
| h2_medium | 10.9158 | None | None | 0.0758 | 0.96 |
| h3_slow | 12.3087 | None | None | 0.0815 | 0.975 |

## Recovered

Brief disturbance followed by a cooling and settling sequence.

- Handle: `exp1_recovered_20260330_131117`
- Tokens: 96
- Coupling: 0.1
- Generation event: exp1_recovered_20260330_131117:000026 @ 2026-03-30 13:11:25
- Readout: y1=0.442838, y2=0.766516, y3=0.334119
- Trajectory: trend=falling, latest=+1.0737, delta=-1.6277, spread=1.6277
- Preview: 'Continuity isn’t a neat line on a graph, a perfectly preserved timeline. It’s the way the scent of your grandmother’s...'

### Output

Continuity isn’t a neat line on a graph, a perfectly preserved timeline. It’s the way the scent of your grandmother’s lilac soap still lingers faintly in your childhood bedroom, a ghost of a fragrance that pulls you back to a specific afternoon of sunlight and whispered stories. It’s the way your hands instinctively reach for a certain worn mug, a familiar weight and warmth that anticipates the quiet comfort of a morning ritual, a feeling you haven’t consciously thought

### Layer Metrics

| Layer | h_norm | entropy | target | saturation | rho |
|---|---:|---:|---:|---:|---:|
| h1_fast | 3.4309 | None | None | 0.0 | 0.94 |
| h2_medium | 11.004 | None | None | 0.0333 | 0.96 |
| h3_slow | 12.4306 | None | None | 0.0883 | 0.975 |
