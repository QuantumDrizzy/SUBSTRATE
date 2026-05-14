/* global React */
const { useEffect: useEffectMV, useState: useStateMV } = React;

function MeshViewport() {
  const [t, setT] = useStateMV(0);
  useEffectMV(() => {
    let raf, start = performance.now();
    const tick = (now) => { setT((now - start) / 1000); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Simple rotating wireframe of a quad mesh, projected
  const rows = 9, cols = 13;
  const a = t * 0.25;
  const cosA = Math.cos(a), sinA = Math.sin(a);
  const project = (x, y, z) => {
    const x2 = x * cosA - z * sinA;
    const z2 = x * sinA + z * cosA;
    const f = 200 / (z2 + 250);
    return { px: 50 + x2 * f * 0.5, py: 50 + (y - z2 * 0.18) * f * 0.45 };
  };
  const pts = [];
  for (let r = 0; r < rows; r++) {
    pts[r] = [];
    for (let c = 0; c < cols; c++) {
      const x = (c - (cols-1)/2) * 14;
      const z = (r - (rows-1)/2) * 14;
      const y = Math.sin((c + r) * 0.6 + t * 1.2) * 6 + Math.cos(c * 0.4 - t) * 4;
      pts[r][c] = project(x, y, z);
    }
  }

  const lines = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols-1; c++) lines.push([pts[r][c], pts[r][c+1]]);
  for (let r = 0; r < rows-1; r++) for (let c = 0; c < cols; c++) lines.push([pts[r][c], pts[r+1][c]]);

  return (
    <div className="grid-bg" style={{position:"absolute",inset:0,overflow:"hidden"}}>
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style={{position:"absolute",inset:0}}>
        <g stroke="var(--cyan-500)" strokeWidth="0.15" opacity="0.85" style={{filter:"drop-shadow(0 0 1px var(--cyan-500))"}}>
          {lines.map(([a,b], i) => <line key={i} x1={a.px} y1={a.py} x2={b.px} y2={b.py} />)}
        </g>
        <g fill="var(--cyan-500)">
          {pts.flat().filter((_,i) => i % 7 === 0).map((p, i) => <circle key={i} cx={p.px} cy={p.py} r="0.3" />)}
        </g>
      </svg>
      <div style={{position:"absolute",left:12,bottom:10,display:"flex",gap:14}} className="lbl">
        <span>θ {(a % (2*Math.PI)).toFixed(3)}rad</span>
        <span>NODES {rows*cols}</span>
        <span>Δt 16.7ms</span>
      </div>
    </div>
  );
}
window.MeshViewport = MeshViewport;
