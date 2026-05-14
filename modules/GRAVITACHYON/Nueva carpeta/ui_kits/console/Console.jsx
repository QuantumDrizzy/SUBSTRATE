/* global React */
const { useState: useStateC, useEffect: useEffectC, useRef: useRefC } = React;

// Initial channel state for the demo
const INITIAL_CHANNELS = [
  { id: "H1:GDS-CALIB_STRAIN",  rate: "16k", status: "locked",  value: "+0.034σ", label: "LOCKED" },
  { id: "L1:GDS-CALIB_STRAIN",  rate: "16k", status: "locked",  value: "+0.029σ", label: "LOCKED" },
  { id: "T0:RC-PHASE_LOCK",     rate: "8k",  status: "nominal", value: "−0.218",  label: "COHERENT" },
  { id: "T0:RC-AMPL",           rate: "8k",  status: "nominal", value: "0.84",    label: "NOMINAL" },
  { id: "L1:SEI-ISOL_X",        rate: "2k",  status: "warn",    value: "+12.4nm", label: "DRIFT" },
  { id: "L1:SEI-ISOL_Y",        rate: "2k",  status: "nominal", value: "+2.1nm",  label: "NOMINAL" },
  { id: "CRYO:STAGE2",          rate: "10",  status: "nominal", value: "4.21K",   label: "NOMINAL" },
  { id: "CRYO:STAGE3",          rate: "10",  status: "nominal", value: "0.92K",   label: "NOMINAL" },
  { id: "VAC:CHAMBER",          rate: "10",  status: "nominal", value: "1.2e−9",  label: "NOMINAL" },
  { id: "LSR:LOCK_NPRO",        rate: "1k",  status: "locked",  value: "STABLE",  label: "LOCKED" },
  { id: "GW:STRAIN_CAL",        rate: "16k", status: "idle",    value: "—",       label: "IDLE" },
  { id: "TS:CURVATURE_K",       rate: "1k",  status: "nominal", value: "+1.4e−24", label: "NOMINAL" },
];

const INITIAL_LOG = [
  { ts: "14:32:08.412Z", level: "info", msg: "H1:GDS-CALIB_STRAIN  σ=+0.034  LOCKED" },
  { ts: "14:32:08.473Z", level: "data", msg: "T0:RC-PHASE_LOCK  ϕ=−0.218  COHERENT" },
  { ts: "14:32:08.534Z", level: "warn", msg: "L1:SEI-ISOL_X  Δ=+12.4nm  DRIFT" },
  { ts: "14:32:08.595Z", level: "info", msg: "CRYO:STAGE2  T=4.21K  NOMINAL" },
  { ts: "14:32:08.656Z", level: "data", msg: "TS:CURVATURE_K  +1.4e−24  NOMINAL" },
  { ts: "14:32:08.718Z", level: "info", msg: "LSR:LOCK_NPRO  PWR=412mW  LOCKED" },
  { ts: "14:32:08.779Z", level: "data", msg: "H1:GDS-CALIB_STRAIN  σ=+0.031  LOCKED" },
];

function Console() {
  const [armed, setArmed]       = useStateC(false);
  const [anomaly, setAnomaly]   = useStateC(false);
  const [paletteOpen, setPalette] = useStateC(false);
  const [focused, setFocused]   = useStateC("H1:GDS-CALIB_STRAIN");
  const [log, setLog]           = useStateC(INITIAL_LOG);
  const [strain, setStrain]     = useStateC(() => Array.from({length: 200}, (_, i) =>
    0.5 + 0.05 * Math.sin(i * 0.3) + 0.03 * Math.sin(i * 0.9)));
  const [readout, setReadout]   = useStateC("+0.0341");
  const [spec, setSpec]         = useStateC(() => buildSpec());

  // Animate strain waveform
  useEffectC(() => {
    let i = strain.length;
    const id = setInterval(() => {
      setStrain(prev => {
        const next = prev.slice(1);
        const base = 0.5 + 0.05 * Math.sin(i * 0.3) + 0.03 * Math.sin(i * 0.9) + (Math.random() - 0.5) * 0.02;
        const v = anomaly ? base + Math.sin(i * 1.4) * 0.25 + (Math.random() - 0.5) * 0.15 : base;
        next.push(Math.max(0.05, Math.min(0.95, v)));
        i++;
        return next;
      });
      setReadout(prev => {
        const f = anomaly ? 4.21 : 0.034 + (Math.random() - 0.5) * 0.004;
        return (anomaly ? "+" : (f >= 0 ? "+" : "")) + f.toFixed(anomaly ? 2 : 4);
      });
    }, 80);
    return () => clearInterval(id);
  }, [anomaly]);

  // Animate spectrogram
  useEffectC(() => {
    const id = setInterval(() => setSpec(prev => stepSpec(prev, anomaly)), 250);
    return () => clearInterval(id);
  }, [anomaly]);

  // Auto-inject anomaly demo
  useEffectC(() => {
    if (!armed) return;
    const id = setTimeout(() => {
      setAnomaly(true);
      setLog(prev => [
        { ts: timeNow(), level: "crit", msg: "H1:GDS-CALIB_STRAIN  σ=+4.21  ANOMALY · 0x4A" },
        ...prev,
      ].slice(0, 7));
    }, 6000);
    return () => clearTimeout(id);
  }, [armed]);

  // ⌘K
  useEffectC(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette(v => !v); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const channels = INITIAL_CHANNELS.map(c => {
    if (anomaly && c.id === "H1:GDS-CALIB_STRAIN") return { ...c, status: "crit", value: "+4.21σ", label: "ANOMALY" };
    if (c.id === "H1:GDS-CALIB_STRAIN") return { ...c, value: readout + "σ" };
    return c;
  });

  return (
    <div className="console" data-screen-label="01 Operator Console">
      <Sidebar channels={channels} focusedId={focused} onFocus={setFocused} />

      {/* Topbar */}
      <header className="console__topbar">
        <img src="../../assets/logo-bug.svg" width="22" height="22" style={{filter:"drop-shadow(0 0 6px rgba(0,240,255,0.4))"}} alt=""/>
        <div>
          <div className="topbar__title">SPACETIME OBSERVATORY</div>
          <div className="topbar__id">SECTOR L1 · OPS DESK · UTC 2026-05-07</div>
        </div>
        <div className="topbar__spacer"/>
        <Button variant="ghost" icon="search" onClick={() => setPalette(true)}>QUERY</Button>
        <span className="topbar__kbd">⌘K</span>
        <Button variant={armed ? "primary" : "default"} onClick={() => { setArmed(true); }}>{armed ? "ARMED" : "ARM"}</Button>
        <Button onClick={() => { setArmed(false); setAnomaly(false); setLog(INITIAL_LOG); }}>RESET</Button>
        <Button variant="warn" onClick={() => { setAnomaly(true); }}>PURGE</Button>
      </header>

      {/* Main grid */}
      <main className="console__main">
        <InstrumentPanel id="3D MESH · SPACETIME CURVATURE" meta="ROT 0.25 rad/s · 117 NODES" status="locked" className="area-mesh">
          <MeshViewport />
        </InstrumentPanel>

        <InstrumentPanel
          id="H1:GDS-CALIB_STRAIN"
          meta="16384 Hz · ±0.002"
          status={anomaly ? "warn" : "locked"}
          armed={armed && !anomaly}
          anomaly={anomaly}
          right={<Badge kind={anomaly ? "warn" : "locked"}>{anomaly ? "ANOMALY · σ=4.21" : "LOCKED"}</Badge>}
          className="area-strain"
        >
          <Readout
            value={readout}
            unit="σ"
            kind={anomaly ? "warn" : "data"}
            sub={`UTC ${timeNow()} · BW 4kHz · 60s avg`}
          />
          <div style={{position:"relative", flex:1, minHeight:0, borderTop:"1px solid var(--line-100)"}}>
            <Oscilloscope data={strain} anomaly={anomaly} />
          </div>
        </InstrumentPanel>

        <InstrumentPanel id="CHANNEL MONITOR" meta={`${channels.length} ACTIVE`} className="area-channels">
          <ChannelTable channels={channels} focusedId={focused} onFocus={setFocused} />
        </InstrumentPanel>

        <InstrumentPanel id="SPECTROGRAM · 0–2 KHZ" meta="60s window · log scale" status="locked" className="area-spec">
          <Spectrogram matrix={spec} />
        </InstrumentPanel>

        <InstrumentPanel id="EVENT LOG · TAIL" meta="3 chan · live" className="area-log">
          <TelemetryLog entries={log} />
        </InstrumentPanel>
      </main>

      <StatusBar armed={armed} anomaly={anomaly} />

      <CommandPalette open={paletteOpen} onClose={() => setPalette(false)} />
    </div>
  );
}

function timeNow() {
  const d = new Date();
  return d.toISOString().slice(11, 23) + "Z";
}

function buildSpec() {
  const ROWS = 24, COLS = 48;
  const m = [];
  for (let r = 0; r < ROWS; r++) {
    m[r] = [];
    for (let c = 0; c < COLS; c++) {
      const band = Math.exp(-Math.pow((r - 14) / 4, 2)) * 0.6;
      m[r][c] = Math.max(0, band + (Math.random() - 0.6) * 0.3);
    }
  }
  return m;
}
function stepSpec(prev, anomaly) {
  const ROWS = prev.length, COLS = prev[0].length;
  const next = prev.map(row => row.slice(1));
  for (let r = 0; r < ROWS; r++) {
    const band = Math.exp(-Math.pow((r - 14) / 4, 2)) * 0.6;
    let v = Math.max(0, band + (Math.random() - 0.6) * 0.3);
    if (anomaly) v += Math.exp(-Math.pow((r - 6) / 3, 2)) * (0.6 + Math.random() * 0.4);
    next[r].push(Math.min(1, v));
  }
  return next;
}

window.Console = Console;
