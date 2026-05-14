# CHRONOS Design System

> **Sovereign Minimalist** — a high-performance interface language for real-time monitoring of spacetime stability, gravitational-wave strain (LIGO-class detectors), and tachyon retrocausal signals. Designed for elite quantum-lab operators who watch the fabric of reality on twelve screens at once.

---

## Product context

CHRONOS is the operator console for a quantum metrology lab. It surfaces:

- **Spacetime stability index** — composite metric across local-frame curvature sensors.
- **Gravitational-wave strain** — LIGO-style interferometer readouts (live h(t), spectrograms, glitch flags).
- **Tachyon retrocausal channels** — speculative signal taps with phase-conjugate coherence scoring.
- **Lab health** — cryostat temps, vacuum, laser lock, seismic isolation.

The single product is a **dark-mode operator console** running 24/7 on multi-monitor workstations. Operators are scientists; the UI must be dense, instantly legible at a glance, and survive long shifts without fatigue. Errors here are catastrophic, so anomalies must read instantly without ever startling.

### Design language: *Sovereign Minimalist*

- **Dark mode absolute** — Deep Obsidian `#08080A`. There is no light mode.
- **Glowing signal accents** — Cyan `#00F0FF` for live data, Warning Orange `#FF9000` for anomalies. Used sparingly; the canvas is mostly negative space.
- **High-density, 40% negative space** — information per square inch is high, but every glyph breathes. Density without clutter.
- **Industrial sharpness** — 1px hairlines, zero corner radius, monospace telemetry, subtle glassmorphism on raised surfaces.
- **Inspiration** — Leica & CERN scientific-instrument typography; SpaceX HUD legibility; Bloomberg terminal density; oscilloscope phosphor persistence.

### Sources

This system was authored from the brief — no codebase, Figma, or screenshots were attached. All visual decisions trace to the design-language paragraph above. If proprietary fonts, logos, or production code exist, drop them in and ask for a re-pass.

---

## Index

| File | Purpose |
|---|---|
| `README.md` | This document. Start here. |
| `colors_and_type.css` | All design tokens — color, type, spacing, motion, elevation. |
| `SKILL.md` | Agent-skill manifest (Claude Code compatible). |
| `fonts/` | (empty — using Google Fonts CDN substitutions; see [Typography](#typography)). |
| `assets/` | Logo marks, signal glyphs, instrument icons. |
| `preview/` | Cards rendered in the Design System tab. |
| `ui_kits/console/` | Operator console UI kit — components + interactive demo. |

---

## Content fundamentals

CHRONOS copy reads like the inside of a scientific instrument. **Terse, instrumented, factual.**

### Voice

- **Third-person observer.** The system never says "I". It rarely says "you" — instead it states the observation. *"Lock acquired"* not *"You acquired lock"*. *"Anomaly at L1:STRAIN_HOLE"* not *"We detected an anomaly"*.
- **Imperative for actions.** Buttons read `ARM`, `LOCK`, `PURGE`, `ACQUIRE`, `HOLD`. No `Click here`, no `Get started`.
- **Numbers always.** A status is a number with units, not a sentiment. `+0.034σ` beats `slightly elevated`. `412 Hz` beats `mid-range`.
- **No hedging.** Never *"approximately"*, *"about"*, *"around"*. Use `±` and a tolerance.

### Casing & punctuation

- **UPPERCASE** for status labels, channel names, action buttons. Tracking is wide (`letter-spacing: 0.16em`) — these are not headings, they are taxonomic labels.
- **Sentence case** for paragraph copy and tooltips.
- **Title Case** is forbidden — it reads commercial and warm; we are neither.
- **Channel identifiers** follow the LIGO convention: `H1:GDS-CALIB_STRAIN`, `T0:RC-PHASE_LOCK`. Always monospace.
- **Timestamps** are ISO-8601 in UTC with subsecond precision: `2026-05-07T14:32:08.412Z`.
- **No exclamation points. No emoji. No ellipses.** Period. (Critical alarms get a glyph and a glow, not a `!!`.)

### Specific phrasings

| Concept | CHRONOS says | Not |
|---|---|---|
| All systems normal | `NOMINAL` / `LOCK STABLE` | "Everything looks great!" |
| A new alert | `ANOMALY · H1:STRAIN · σ=4.2` | "Heads up — something's off" |
| Loading | `ACQUIRING…` (mono, no animation beyond a phosphor blink) | spinner |
| Empty state | `NO SIGNAL · CHANNEL IDLE` | "Nothing to see here yet" |
| Error | `LOCK LOST · 14:32:08.412Z · CODE 0x4A` | "Something went wrong" |
| Confirmation | `ARMED` / `ACQUIRED` / `PURGED` | "Done!" / "Success" |

### Tone vibe

The tone is **calm, technical, slightly austere — but never cold to its operator**. Tooltips and inline help may briefly explain physics in plain language (*"Strain ratio between the X- and Y-arms of the interferometer"*), but the chrome itself is dispassionate. The operator is a peer; the system is a precision instrument they trust.

---

## Visual foundations

### Color

The palette is monochromatic obsidian + two glowing signal channels. Auxiliary colors exist but are reserved for specific physical signals (magenta = tachyon, violet = gravitational, green = nominal lock, red = critical alarm). Do not invent colors.

- **Background** is always `#08080A`. Panels lift via 1-step lighter obsidian (`#0C0C10`), not via shadow.
- **Cyan `#00F0FF`** marks live data — sparklines, active values, focused inputs, primary action edges.
- **Orange `#FF9000`** marks anomalies and warnings. Never use orange decoratively.
- **Glow** is achieved with `text-shadow` and `box-shadow` using the channel color at ~45% alpha. Glow is the *only* form of emphasis; never use bold colored fills for backgrounds.

### Typography

Three families, each with a specific job:

- **JetBrains Mono** — telemetry, channel IDs, timestamps, numeric readouts. ~70% of all text on screen.
- **Inter Tight** — UI chrome, buttons, paragraphs, labels.
- **Space Grotesk** — display headings only (screen titles, instrument names).

Type is **small, tight, and tracked wide for labels.** Body text is 14px; labels are 11px uppercase with `0.16em` tracking; readouts are 28px tabular monospace. There are no decorative weights — 300 / 400 / 500 / 600 only.

> **Font substitution flag** — JetBrains Mono, Inter Tight, and Space Grotesk are loaded from Google Fonts CDN as stand-ins. If CHRONOS has proprietary type (e.g. a CERN-licensed cut, a custom mono), please drop the files in `fonts/` and we'll switch the `@font-face` rules.

### Spacing & layout

- 4px base grid. Most spacing tokens are 4 / 8 / 16 / 24 / 32 / 48.
- Panels are 1px-bordered rectangles flush with each other; **gutters are exactly 1px** (a hairline) on tightly coupled instruments, or 24px on logically separate clusters. Never anything in between.
- **40% negative-space rule:** if you measure a screen and the dark substrate doesn't account for at least 40% of the pixels, it is too dense. Prune.
- Layouts are grid-pinned. Drag-resizable panels are allowed; free-floating windows are not.

### Backgrounds

- The default background is solid `#08080A`. No gradients, no images, no textures.
- Consoles may layer a **32px square hairline grid** at 6% alpha as a backdrop for vector viewports (3D mesh, oscilloscopes). Grid is structural, not decorative.
- A subtle **scanline** (1px horizontal, 2% alpha, repeating every 3px) may be added to oscilloscope panels only — phosphor authenticity, not a global effect.

### Borders & dividers

- Borders are always **1px** and always one of four hairline tokens: `--line-100/200/300/400` (6% / 10% / 16% / 24% alpha white).
- The default panel border is `--line-200`. Hover lifts to `--line-300`. Focused/armed instruments get a full-channel-color border (`#00F0FF` or `#FF9000`).
- Dividers between rows in a table use `--line-100` — barely visible, present.

### Corner radius

**Zero. No exceptions** — except status dots (4px circles) which use `border-radius: 50%`. Buttons, inputs, panels, modals, badges: all square.

### Shadows & elevation

There is no traditional drop-shadow language — flat surfaces, lifted by 1px borders and ambient glow.

- **Inset 1px hairline** = panel.
- **Inset 1px hairline + 1px black bottom** = raised surface (button, input).
- **0 24px 48px -12px black + inset hairline** = modal overlay.
- **Channel-color glow** (e.g. `0 0 12px var(--cyan-glow)`) marks an *armed* or *focused* element only. Glow is information, not decoration.

### Glassmorphism

Used sparingly: modal overlays, popovers, the command palette. Always:

```css
background: rgba(12, 12, 16, 0.72);
backdrop-filter: blur(16px) saturate(140%);
border: 1px solid var(--line-300);
```

Never on primary panels — operators need predictable contrast, not pretty.

### Hover & press

- **Hover (interactive surface):** background steps from `--obsidian-100` → `--obsidian-200`. Border steps from `--line-200` → `--line-300`. Duration 80ms.
- **Hover (text/icon button):** color steps from `--fg-muted` → `--fg`. No background change.
- **Press:** background steps to `--obsidian-300`, **scale: 0.99** (a 1% mechanical click), 80ms. Never bounce, never overshoot.
- **Focus:** 1px cyan ring (`box-shadow: 0 0 0 1px var(--cyan-500)`) — no soft glow on focus, that's reserved for armed/active.
- **Armed/active:** full cyan border + soft outer cyan glow. This is the *engaged* state.

### Motion

Fast and mechanical. The instrument feels rigid.

- **Durations:** 80ms (hover), 140ms (panel open / value change), 260ms (modal mount). Nothing slower.
- **Easing:** `cubic-bezier(0.2, 0, 0.1, 1)` standard, `cubic-bezier(0.6, 0, 0.4, 1)` for snap-to-value transitions. **No bounce, no overshoot.**
- **Number changes** crossfade through digit slots (tabular-nums); they never tick or animate the value itself.
- **Phosphor blink** — when a new sample arrives on a live channel, the channel border pulses cyan once over 200ms. This is the only ambient animation.

### Imagery

The product itself contains **no photographic imagery.** Visualizations are vector: 3D wireframe meshes, oscilloscope traces, spectrogram heatmaps, polar plots. If marketing material requires imagery, it should be **monochrome high-contrast lab photography** with cyan or orange grading — never warm, never saturated.

### Transparency & blur

- Solid backgrounds for primary panels.
- Translucent (`rgba(12,12,16,0.72)` + `backdrop-filter: blur(16px)`) for modals, popovers, command palette only.
- Plot overlays may use 8% alpha cyan/orange washes to fill area-under-curve.

### Cards

In CHRONOS, "cards" are **instrument panels**: a 1px-bordered rectangle, an uppercase mono label in the top-left corner, sometimes a status dot, the data filling the rest. No padding inside the border on the data side; padding only around the label/header strip. They tile flush with each other.

---

## Iconography

CHRONOS uses **Lucide** ([lucide.dev](https://lucide.dev)) as its icon system — thin 1.5px stroke, square caps, no fills. Lucide is loaded from CDN; specific glyphs we rely on are cached as SVGs in `assets/icons/` for offline operator stations.

### Rules

- **Stroke weight is always 1.5px.** Never fill icons. Never mix stroke widths on one screen.
- **Color is `currentColor`** so icons inherit `--fg-muted` by default and channel colors when placed inside an armed control.
- **Size grid:** 14 / 16 / 20 / 24px only. Most UI icons are 16px.
- **Domain-specific glyphs** (interferometer schematic, cryostat, phase-lock loop) are custom SVGs, also 1.5px stroke, drawn to match Lucide's geometric language. They live in `assets/icons/instruments/`.

### Emoji & unicode

- **Emoji: never.** Operators don't get pictographs from a phone keyboard.
- **Unicode glyphs** are used for math and physics: `σ`, `λ`, `Δ`, `±`, `→`, `·`, `…` (the last is forbidden in copy but allowed as a typographic dot leader). Greek letters are rendered in JetBrains Mono.

### Logo

`assets/logo.svg` is the wordmark — `CHRONOS` set in Space Grotesk Medium with the "O" replaced by a circular interferometer schematic glyph. The bug-only variant `assets/logo-bug.svg` is the schematic glyph alone, used as a favicon and as the top-left-corner mark on the console.

> **Asset substitution flag** — the logo is generated from the brief, not received from the brand. If a real CHRONOS logo exists, drop it in `assets/` and we'll re-cap.

---

## Iterating

Tell the agent:

> "Use the CHRONOS design system to mock up a [thing]."

It'll read `colors_and_type.css`, copy logos and icons, and produce HTML that looks like it came out of a vacuum chamber.
