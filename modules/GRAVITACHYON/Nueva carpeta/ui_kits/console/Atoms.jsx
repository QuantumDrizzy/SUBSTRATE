/* global React */
const { useState, useEffect, useRef } = React;

// ===== Atoms =====
function Icon({ name, size = 16, color = "currentColor", strokeWidth = 1.5 }) {
  const paths = {
    play:    <polygon points="5,3 17,10 5,17"/>,
    stop:    <rect x="5" y="5" width="10" height="10"/>,
    plus:    <g><line x1="4" y1="10" x2="16" y2="10"/><line x1="10" y1="4" x2="10" y2="16"/></g>,
    search:  <g><circle cx="9" cy="9" r="6"/><line x1="13" y1="13" x2="17" y2="17"/></g>,
    cmd:     <path d="M6 4 H14 M6 16 H14 M4 6 V14 M16 6 V14 M4 4 H6 V6 M14 4 H16 V6 M4 14 H6 V16 M14 14 H16 V16"/>,
    target:  <g><circle cx="10" cy="10" r="7"/><line x1="10" y1="2" x2="10" y2="5"/><line x1="10" y1="15" x2="10" y2="18"/><line x1="2" y1="10" x2="5" y2="10"/><line x1="15" y1="10" x2="18" y2="10"/></g>,
    bolt:    <polyline points="11,2 4,11 9,11 7,18 14,9 9,9 11,2"/>,
    waves:   <path d="M2 10 Q5 5, 8 10 T 14 10 T 20 10"/>,
    pulse:   <polyline points="2,10 6,10 8,4 12,16 14,10 18,10"/>,
    grid:    <g><rect x="3" y="3" width="6" height="6"/><rect x="11" y="3" width="6" height="6"/><rect x="3" y="11" width="6" height="6"/><rect x="11" y="11" width="6" height="6"/></g>,
    cog:     <g><circle cx="10" cy="10" r="3"/><path d="M10 2 V4 M10 16 V18 M2 10 H4 M16 10 H18 M4.4 4.4 L5.8 5.8 M14.2 14.2 L15.6 15.6 M4.4 15.6 L5.8 14.2 M14.2 5.8 L15.6 4.4"/></g>,
    log:     <g><line x1="4" y1="5" x2="16" y2="5"/><line x1="4" y1="10" x2="16" y2="10"/><line x1="4" y1="15" x2="12" y2="15"/></g>,
    arrow:   <g><line x1="3" y1="10" x2="17" y2="10"/><polyline points="13,6 17,10 13,14"/></g>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke={color}
         strokeWidth={strokeWidth} strokeLinecap="square" style={{flex:"none"}}>
      {paths[name] || null}
    </svg>
  );
}

function StatusDot({ kind = "idle" }) {
  return <span className={`dot dot--${kind}`} />;
}

function Badge({ kind = "default", children }) {
  const colorMap = {
    nominal: "var(--green-500)", locked: "var(--cyan-500)",
    warn: "var(--orange-500)", crit: "var(--red-500)",
    idle: "var(--fg-faint)", default: "var(--fg-muted)",
  };
  const glow = { locked: "0 0 8px rgba(0,240,255,0.45)", warn: "0 0 8px rgba(255,144,0,0.45)", crit: "0 0 8px rgba(255,51,68,0.4)" };
  return <span className="badge" style={{ color: colorMap[kind], boxShadow: glow[kind] || "none" }}>{children}</span>;
}

function Button({ variant = "default", icon, children, ...props }) {
  const cls = "btn" + (variant !== "default" ? ` btn--${variant}` : "") + (icon && !children ? " btn--icon" : "");
  return <button className={cls} {...props}>{icon && <Icon name={icon} size={14} />}{children}</button>;
}

Object.assign(window, { Icon, StatusDot, Badge, Button });
