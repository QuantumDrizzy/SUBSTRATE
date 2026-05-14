/* global React */
window.AnomalyToast = function AnomalyToast({ anomaly, onDismiss, onInvestigate }) {
  if (!anomaly) return null;
  return (
    <div className="chr-toast">
      <div className="chr-toast__bar" />
      <div className="chr-toast__body">
        <div className="chr-toast__head">
          <span className="chr-toast__lvl">ANOMALY</span>
          <span className="chr-toast__ts">{anomaly.ts}</span>
        </div>
        <div className="chr-toast__chan">{anomaly.channel}</div>
        <div className="chr-toast__msg">{anomaly.message}</div>
        <div className="chr-toast__actions">
          <window.Button variant="warn" onClick={onInvestigate}>INVESTIGATE</window.Button>
          <window.Button variant="ghost" onClick={onDismiss}>DISMISS</window.Button>
        </div>
      </div>
    </div>
  );
};
