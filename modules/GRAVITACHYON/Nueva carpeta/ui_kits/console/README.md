# CHRONOS Console — UI Kit

The operator console: the canonical CHRONOS surface. A multi-panel HUD optimised for 24/7 wallboard duty, displaying live spacetime stability, gravitational-wave strain, tachyon retrocausal channels, and lab health.

## Files

| File | Purpose |
|---|---|
| `index.html` | Interactive demo. Open it. |
| `Console.jsx` | Top-level layout (sidebar + grid + status bar). |
| `Sidebar.jsx` | Channel list, navigation, brand mark. |
| `StatusBar.jsx` | Bottom UTC clock + lab-health pills. |
| `InstrumentPanel.jsx` | Reusable bordered panel (header + body slot). |
| `Readout.jsx` | Numeric readouts with channel-color glow. |
| `Oscilloscope.jsx` | Animated phosphor-persistence trace. |
| `Spectrogram.jsx` | Live spectrogram heatmap. |
| `MeshViewport.jsx` | 3D vector mesh on hairline grid. |
| `TelemetryLog.jsx` | Scrolling color-coded log. |
| `ChannelTable.jsx` | Tabular channel monitor with status dots. |
| `CommandPalette.jsx` | ⌘K palette overlay (glassmorphic). |
| `Button.jsx`, `StatusDot.jsx`, `Badge.jsx`, `Icon.jsx` | Atoms. |

## How to use it

```html
<script type="text/babel" src="Icon.jsx"></script>
<script type="text/babel" src="Button.jsx"></script>
<script type="text/babel" src="StatusDot.jsx"></script>
<script type="text/babel" src="Badge.jsx"></script>
<script type="text/babel" src="InstrumentPanel.jsx"></script>
<!-- … -->
<script type="text/babel" src="Console.jsx"></script>
```

Each component exports itself onto `window` for cross-script use.

## Interactive behaviours

- **⌘K** opens the command palette.
- Click a sidebar channel to focus it (cyan border).
- Click `ARM` / `LOCK` / `PURGE` in the top bar to flip the demo's lab state.
- The `H1` strain readout drifts continuously. After ~6 seconds it injects a synthetic anomaly that reads orange + glow.

These are demo-fidelity interactions. They are not connected to a real interferometer.
