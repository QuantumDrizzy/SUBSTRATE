/* global React */
const { useState } = React;

window.Button = function Button({ variant = "default", icon, children, onClick, disabled }) {
  const cls = ["chr-btn", `chr-btn--${variant}`, disabled && "chr-btn--disabled", icon && !children && "chr-btn--icon"].filter(Boolean).join(" ");
  return (
    <button className={cls} onClick={disabled ? undefined : onClick} disabled={disabled}>
      {icon ? <span className="chr-btn__icon">{icon}</span> : null}
      {children ? <span>{children}</span> : null}
    </button>
  );
};

window.StatusBadge = function StatusBadge({ status = "nominal", children }) {
  const map = {
    nominal: { color: "var(--nominal)", glow: false, label: "NOMINAL" },
    locked:  { color: "var(--cyan-500)", glow: true,  label: "LOCKED"  },
    anomaly: { color: "var(--orange-500)", glow: true,  label: "ANOMALY" },
    critical:{ color: "var(--critical)", glow: true, label: "CRITICAL"},
    idle:    { color: "var(--fg-400)", glow: false, label: "IDLE"    },
  };
  const m = map[status] || map.nominal;
  return (
    <span
      className="chr-badge"
      style={{
        color: m.color,
        boxShadow: m.glow ? `0 0 8px ${m.color}66` : "none",
      }}
    >
      {children || m.label}
    </span>
  );
};

window.StatusDot = function StatusDot({ status = "nominal" }) {
  const colorMap = {
    nominal: "var(--nominal)",
    locked:  "var(--cyan-500)",
    anomaly: "var(--orange-500)",
    critical:"var(--critical)",
    idle:    "var(--fg-500)",
  };
  const c = colorMap[status] || "var(--fg-500)";
  return (
    <span
      className="chr-dot"
      style={{
        background: c,
        boxShadow: status === "idle" ? "none" : `0 0 6px ${c}`,
      }}
    />
  );
};
