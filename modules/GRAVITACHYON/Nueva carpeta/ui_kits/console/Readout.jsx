/* global React */
function Readout({ value, unit, sub, kind = "data" }) {
  const cls = "readout" + (kind === "warn" ? " readout--warn" : kind === "nom" ? " readout--nom" : "");
  return (
    <div className="readout-block">
      <div className={cls}>
        {value}
        {unit && <span className="readout__unit">{unit}</span>}
      </div>
      {sub && <div className="readout__sub">{sub}</div>}
    </div>
  );
}
window.Readout = Readout;
