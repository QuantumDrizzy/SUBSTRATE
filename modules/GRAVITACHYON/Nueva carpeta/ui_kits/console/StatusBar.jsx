/* global React */
const { useEffect: useEffectS, useState: useStateS } = React;

function StatusBar({ armed, anomaly }) {
  const [now, setNow] = useStateS(new Date());
  useEffectS(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const iso = now.toISOString().replace("T", " ").replace("Z", " UTC");
  return (
    <footer className="console__statusbar">
      <span>{iso}</span>
      <span style={{color: armed ? "var(--cyan-500)" : "var(--fg-faint)"}}>● {armed ? "ARMED" : "STANDBY"}</span>
      <span style={{color:"var(--green-500)"}}>● LOCK STABLE</span>
      <span style={{color: anomaly ? "var(--orange-500)" : "var(--fg-faint)"}}>● {anomaly ? "ANOMALY · σ=4.21" : "NO ANOMALIES"}</span>
      <span style={{flex:1}}/>
      <span>CRYO 4.21K</span>
      <span>VAC 1.2e−9 mbar</span>
      <span>SEI 2.1nm RMS</span>
      <span style={{color:"var(--fg-muted)"}}>BUILD 4.182.07</span>
    </footer>
  );
}
window.StatusBar = StatusBar;
