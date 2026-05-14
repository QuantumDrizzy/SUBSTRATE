/* global React */
function TelemetryLog({ entries }) {
  return (
    <div className="log">
      {entries.map((e, i) => (
        <div className="log__row" key={i}>
          <span className="log__ts">{e.ts}</span>
          <span className={`log__lvl--${e.level}`}>{e.level.toUpperCase()}</span>
          <span className="log__msg">{e.msg}</span>
        </div>
      ))}
    </div>
  );
}
window.TelemetryLog = TelemetryLog;
