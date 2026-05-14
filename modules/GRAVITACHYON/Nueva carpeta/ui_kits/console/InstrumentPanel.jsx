/* global React */
function InstrumentPanel({ id, meta, status = "nominal", armed, anomaly, right, children, className = "" }) {
  const cls = "panel " + (armed ? "panel--armed " : "") + (anomaly ? "panel--anomaly " : "") + className;
  return (
    <div className={cls.trim()}>
      <div className="panel__head">
        <div className="panel__title">
          <StatusDot kind={anomaly ? "warn" : status} />
          <span className="panel__id">{id}</span>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          {meta && <span className="panel__meta">{meta}</span>}
          {right}
        </div>
      </div>
      <div className="panel__body">{children}</div>
    </div>
  );
}
window.InstrumentPanel = InstrumentPanel;
