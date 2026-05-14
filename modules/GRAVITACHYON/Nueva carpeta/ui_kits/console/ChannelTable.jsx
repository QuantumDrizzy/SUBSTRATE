/* global React */
function ChannelTable({ channels, focusedId, onFocus }) {
  return (
    <div style={{height:"100%", overflow:"auto"}}>
      <table className="chtable">
        <thead><tr><th></th><th>Channel</th><th style={{textAlign:"right"}}>Value</th><th>Status</th></tr></thead>
        <tbody>
          {channels.map(c => (
            <tr key={c.id} onClick={() => onFocus?.(c.id)}
                style={{background: c.id === focusedId ? "rgba(0,240,255,0.05)" : "transparent"}}>
              <td><StatusDot kind={c.status}/></td>
              <td style={{color: c.id === focusedId ? "var(--cyan-500)" : "var(--fg-muted)"}}>{c.id}</td>
              <td className="value">{c.value}</td>
              <td><span style={{color:
                c.status==="nominal"?"var(--green-500)":
                c.status==="locked"?"var(--cyan-500)":
                c.status==="warn"?"var(--orange-500)":
                c.status==="crit"?"var(--red-500)":"var(--fg-faint)"}}>{c.label}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
window.ChannelTable = ChannelTable;
