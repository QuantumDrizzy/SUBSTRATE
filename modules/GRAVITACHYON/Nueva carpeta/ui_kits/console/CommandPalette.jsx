/* global React */
const { useEffect: useEffectCP, useState: useStateCP } = React;

function CommandPalette({ open, onClose }) {
  const [q, setQ] = useStateCP("");
  useEffectCP(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  const cmds = [
    { id: "ARM",     hint: "↵", icon: "bolt" },
    { id: "LOCK",    hint: "⌘L", icon: "target" },
    { id: "PURGE",   hint: "⌘⇧P", icon: "stop" },
    { id: "ACQUIRE H1:STRAIN", hint: "", icon: "pulse" },
    { id: "ACQUIRE T0:RC-PHASE", hint: "", icon: "waves" },
    { id: "OPEN EVENT LOG", hint: "⌘E", icon: "log" },
    { id: "SETTINGS", hint: "⌘,", icon: "cog" },
  ].filter(c => c.id.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="palette-backdrop" onClick={onClose}>
      <div className="palette" onClick={e => e.stopPropagation()}>
        <input
          className="palette__input"
          autoFocus
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="QUERY · CHANNEL · COMMAND"
        />
        <div className="palette__list">
          {cmds.map((c, i) => (
            <div key={c.id} className={"palette__row" + (i === 0 ? " palette__row--active" : "")}>
              <Icon name={c.icon} size={14} />
              <span>{c.id}</span>
              <span className="palette__row__hint">{c.hint}</span>
            </div>
          ))}
          {cmds.length === 0 && <div className="palette__row" style={{color:"var(--fg-faint)"}}>NO MATCH</div>}
        </div>
      </div>
    </div>
  );
}
window.CommandPalette = CommandPalette;
