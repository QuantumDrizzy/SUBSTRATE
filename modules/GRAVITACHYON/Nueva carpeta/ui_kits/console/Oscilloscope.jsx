/* global React */
const { useEffect, useRef, useState } = React;

function Oscilloscope({ data, anomaly }) {
  const wrap = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 200 });
  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => {
      const r = e.contentRect;
      setSize({ w: Math.max(100, r.width), h: Math.max(60, r.height) });
    });
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const { w, h } = size;
  const N = data.length;
  const stepX = w / (N - 1);
  const path = data.map((v, i) => `${i === 0 ? "M" : "L"}${(i*stepX).toFixed(1)},${((1-v)*h).toFixed(1)}`).join(" ");
  const color = anomaly ? "var(--orange-500)" : "var(--cyan-500)";

  return (
    <div ref={wrap} className="grid-bg" style={{position:"absolute",inset:0,overflow:"hidden"}}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{position:"absolute",inset:0}}>
        <line x1="0" y1={h/2} x2={w} y2={h/2} stroke="var(--line-200)" strokeDasharray="2 4" />
        <path d={path} stroke={color} strokeWidth="1.4" fill="none" className={anomaly ? "scope-trace scope-trace--warn" : "scope-trace"} />
        <path d={path} stroke={color} strokeWidth="2.5" fill="none" className="scope-tail" />
      </svg>
    </div>
  );
}
window.Oscilloscope = Oscilloscope;
