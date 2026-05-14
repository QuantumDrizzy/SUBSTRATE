/* global React */
function Sidebar({ channels, focusedId, onFocus }) {
  return (
    <aside className="console__sidebar">
      <div className="brand">
        <img className="brand__bug" src="../../assets/logo-bug.svg" alt="" />
        <span className="brand__name">CHRONOS</span>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section__title">Instruments</div>
        {[
          {n:"OVERVIEW", icon:"grid", active:true},
          {n:"INTERFEROMETER", icon:"target"},
          {n:"TACHYON ARRAY", icon:"bolt"},
          {n:"CRYO STACK", icon:"waves"},
          {n:"EVENT LOG", icon:"log"},
          {n:"SETTINGS", icon:"cog"},
        ].map(r => (
          <div key={r.n} className={"nav-row" + (r.active ? " nav-row--active" : "")}>
            <Icon name={r.icon} size={14} />
            <span className="nav-row__id">{r.n}</span>
          </div>
        ))}
      </div>

      <div className="sidebar-section" style={{flex:1, overflow:"auto", minHeight:0}}>
        <div className="sidebar-section__title">Channels · {channels.length}</div>
        {channels.map(c => (
          <div key={c.id}
               className={"nav-row" + (c.id === focusedId ? " nav-row--active" : "")}
               onClick={() => onFocus?.(c.id)}>
            <StatusDot kind={c.status} />
            <span className="nav-row__id">{c.id}</span>
            <span className="nav-row__rate">{c.rate}</span>
          </div>
        ))}
      </div>

      <div className="sidebar-foot">
        <span>OPERATOR · K. VOSS</span>
        <span style={{color:"var(--green-500)"}}>● LINK</span>
      </div>
    </aside>
  );
}
window.Sidebar = Sidebar;
