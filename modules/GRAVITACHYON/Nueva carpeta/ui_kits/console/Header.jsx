/* global React */
const { useEffect: useEffectHd, useState: useStateHd } = React;

function fmtClock() {
  const d = new Date();
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss} UTC`;
}

window.Header = function Header({ session, operator, onPaletteOpen }) {
  const [clock, setClock] = useStateHd(fmtClock);
  useEffectHd(() => {
    const id = setInterval(() => setClock(fmtClock()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <header className="chr-header">
      <div className="chr-header__left">
        <img src="../../assets/logo-bug.svg" className="chr-header__bug" alt="" />
        <span className="chr-header__wordmark">CHRONOS</span>
        <span className="chr-header__divider" />
        <span className="chr-header__crumb">OPERATOR · CONSOLE</span>
        <span className="chr-header__crumb chr-header__crumb--muted">/ SESSION {session}</span>
      </div>
      <div className="chr-header__center">
        <button className="chr-cmd" onClick={onPaletteOpen}>
          <svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="9" cy="9" r="6" /><line x1="13" y1="13" x2="17" y2="17" /></svg>
          <span>QUERY CHANNEL · COMMAND</span>
          <span className="chr-cmd__kbd">⌘K</span>
        </button>
      </div>
      <div className="chr-header__right">
        <span className="chr-header__clock">{clock}</span>
        <span className="chr-header__divider" />
        <span className="chr-header__op">
          <span className="chr-header__op-dot" />
          {operator}
        </span>
      </div>
    </header>
  );
};
