/* global React */
function Spectrogram({ matrix }) {
  // matrix: 2D array of intensities [0..1], rows = freq bins (top = high), cols = time
  const rows = matrix.length, cols = matrix[0].length;
  const cellW = 100 / cols, cellH = 100 / rows;
  return (
    <div style={{position:"absolute",inset:0,padding:"0 32px 18px 36px"}}>
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style={{display:"block"}}>
        {matrix.map((row, r) => row.map((v, c) => {
          const a = Math.min(1, Math.max(0, v));
          const hue = a > 0.7 ? "255,144,0" : "0,240,255";
          return <rect key={r+"-"+c} x={c*cellW} y={r*cellH} width={cellW+0.3} height={cellH+0.3} fill={`rgba(${hue},${a.toFixed(2)})`} />;
        }))}
      </svg>
      {/* axes */}
      <svg className="spec-axis" width="100%" height="14" style={{position:"absolute",left:0,bottom:0}} viewBox="0 0 100 14" preserveAspectRatio="none">
        <text x="2" y="10">−60s</text><text x="48" y="10">−30s</text><text x="92" y="10">NOW</text>
      </svg>
      <svg className="spec-axis" width="32" height="100%" style={{position:"absolute",right:0,top:0,height:"calc(100% - 18px)"}} viewBox="0 0 32 100" preserveAspectRatio="none">
        <text x="2" y="6">2kHz</text><text x="2" y="50">1kHz</text><text x="2" y="98">0Hz</text>
      </svg>
    </div>
  );
}
window.Spectrogram = Spectrogram;
