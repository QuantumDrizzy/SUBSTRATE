#pragma once
#include <QString>

// QSS stylesheet derived 1:1 from the SUBSTRATE design system.
// Colors, radii, spacing, type scale — all match colors_and_type.css exactly.
//
// Light palette is the default. Dark mode swaps surface tokens via a separate
// sheet appended at runtime when the system palette is dark.

namespace Theme {

// ── Color tokens (light) ────────────────────────────────────────────────────
static constexpr auto BASE          = "#f5f4f1";
static constexpr auto SURFACE       = "#fffefb";
static constexpr auto SURFACE_2     = "#ebe9e3";
static constexpr auto SURFACE_3     = "#e1ded6";
static constexpr auto GLASS         = "rgba(255,254,251,220)";  // ~0.86 alpha (QColor range 0-255)
static constexpr auto GLASS_STRONG  = "rgba(255,254,251,245)";

static constexpr auto FG            = "#1c1b19";
static constexpr auto FG_2          = "#4a4742";
static constexpr auto FG_3          = "#7a766e";
static constexpr auto FG_4          = "#a8a39a";

static constexpr auto BORDER        = "#d8d4cb";
static constexpr auto BORDER_SUBTLE = "#e6e2d9";
static constexpr auto BORDER_STRONG = "#b9b3a6";

static constexpr auto SCORE_HIGH    = "#2f7a4d";
static constexpr auto SCORE_HIGH_BG = "#e3eee5";
static constexpr auto SCORE_MED     = "#a07020";
static constexpr auto SCORE_MED_BG  = "#f3ead6";
static constexpr auto SCORE_LOW     = "#a8392f";
static constexpr auto SCORE_LOW_BG  = "#f0dfdb";
static constexpr auto SCORE_IDLE    = "#8a857b";
static constexpr auto SCORE_IDLE_BG = "#e8e5dd";

static constexpr auto FOCUS         = "#b08a3c";
static constexpr auto SELECTION_BG  = "#eee9d9";  // row selected highlight

// ── Dark palette tokens ──────────────────────────────────────────────────────
static constexpr auto DARK_BASE         = "#1c1b19";
static constexpr auto DARK_SURFACE      = "#25231f";
static constexpr auto DARK_SURFACE_2    = "#181715";
static constexpr auto DARK_FG           = "#ece9e2";
static constexpr auto DARK_FG_2         = "#b8b3a8";
static constexpr auto DARK_FG_3         = "#8a857b";
static constexpr auto DARK_BORDER       = "#34322d";
static constexpr auto DARK_BORDER_SUB   = "#2a2824";
static constexpr auto DARK_SCORE_HIGH   = "#6ec288";
static constexpr auto DARK_SCORE_MED    = "#d4a35a";
static constexpr auto DARK_SCORE_LOW    = "#d97a6f";

// ── Font stacks ──────────────────────────────────────────────────────────────
// Qt resolves the first matching installed family.
static constexpr auto FONT_UI   = "Inter Tight";
static constexpr auto FONT_MONO = "JetBrains Mono";
static constexpr auto FONT_UI_FALLBACK   = "Segoe UI";
static constexpr auto FONT_MONO_FALLBACK = "Consolas";

// ── Base QSS ────────────────────────────────────────────────────────────────
inline QString lightSheet()
{
    return QStringLiteral(R"QSS(

/* ── App shell ──────────────────────────────────────────────────────────── */
QMainWindow, QWidget#AppRoot {
    background: #f5f4f1;
}

/* ── Toolbar ────────────────────────────────────────────────────────────── */
QWidget#Toolbar {
    background: rgba(255,254,251,220);
    border-bottom: 1px solid #d8d4cb;
    min-height: 40px;
    max-height: 40px;
}
QLabel#Wordmark {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    color: #1c1b19;
}
QLabel#ToolbarStatus {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #4a4742;
    background: #fffefb;
    border: 1px solid #d8d4cb;
    border-radius: 4px;
    padding: 0 10px;
    min-height: 26px;
    max-height: 26px;
}

/* ── Toolbar icon buttons ────────────────────────────────────────────────── */
QPushButton#IconBtn {
    background: transparent;
    border: none;
    border-radius: 4px;
    color: #4a4742;
    min-width: 28px;  max-width: 28px;
    min-height: 28px; max-height: 28px;
    padding: 0;
    font-size: 14px;
}
QPushButton#IconBtn:hover {
    background: #ebe9e3;
    color: #1c1b19;
}
QPushButton#IconBtn:pressed {
    background: #e1ded6;
}
QPushButton#IconBtn:checked {
    background: #eee9d9;
    color: #1c1b19;
}

/* ── Sidebar / Inspector (glass panels) ─────────────────────────────────── */
QWidget#Sidebar, QWidget#Inspector {
    background: rgba(255,254,251,220);
    border: 1px solid #d8d4cb;
    border-radius: 6px;
}
QLabel#PanelEyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #7a766e;
    text-transform: uppercase;
}
QLabel#PanelHeading {
    font-size: 14px;
    font-weight: 600;
    color: #1c1b19;
}

/* ── Search input ────────────────────────────────────────────────────────── */
QLineEdit#SearchInput {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
    background: #ebe9e3;
    border: 1px solid #d8d4cb;
    border-radius: 4px;
    padding: 0 6px;
    min-height: 22px;
    max-height: 22px;
}
QLineEdit#SearchInput:focus {
    border-color: #b08a3c;
    outline: none;
}
QLineEdit#SearchInput::placeholder {
    color: #a8a39a;
}

/* ── Layer list rows ─────────────────────────────────────────────────────── */
QWidget#LayerRow {
    border-radius: 4px;
    min-height: 28px;
    max-height: 28px;
}
QWidget#LayerRow:hover {
    background: #ebe9e3;
}
QWidget#LayerRow[selected="true"] {
    background: #eee9d9;
}
QLabel#LayerName {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
}
QLabel#LayerScore {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    font-variant-numeric: tabular-nums;
}

/* ── Main content panels ─────────────────────────────────────────────────── */
QFrame#ContentPanel {
    background: #fffefb;
    border: 1px solid #d8d4cb;
    border-radius: 6px;
}
QLabel#PanelTitle {
    font-size: 14px;
    font-weight: 600;
    color: #1c1b19;
}

/* ── Score bar track ─────────────────────────────────────────────────────── */
QProgressBar#ScoreTrack {
    background: #ebe9e3;
    border: none;
    border-radius: 2px;
    min-height: 8px;
    max-height: 8px;
    text-align: right;
}
QProgressBar#ScoreTrack::chunk {
    border-radius: 2px;
}

/* ── Status pills ────────────────────────────────────────────────────────── */
QLabel#PillHigh {
    background: #e3eee5; color: #2f7a4d;
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px;
    border-radius: 999px;
    padding: 2px 8px;
    min-height: 18px; max-height: 18px;
}
QLabel#PillMed {
    background: #f3ead6; color: #a07020;
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px;
    border-radius: 999px;
    padding: 2px 8px;
    min-height: 18px; max-height: 18px;
}
QLabel#PillLow {
    background: #f0dfdb; color: #a8392f;
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px;
    border-radius: 999px;
    padding: 2px 8px;
    min-height: 18px; max-height: 18px;
}
QLabel#PillIdle {
    background: #e8e5dd; color: #8a857b;
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px;
    border-radius: 999px;
    padding: 2px 8px;
    min-height: 18px; max-height: 18px;
}

/* ── Large numeric readout ───────────────────────────────────────────────── */
QLabel#Readout {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 22px;
    font-weight: 500;
    color: #1c1b19;
}

/* ── Key-value rows ──────────────────────────────────────────────────────── */
QLabel#KVKey {
    font-size: 11px;
    color: #7a766e;
}
QLabel#KVVal {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
}

/* ── Data inputs ─────────────────────────────────────────────────────────── */
QLineEdit#DataInput {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
    background: #ebe9e3;
    border: 1px solid #d8d4cb;
    border-radius: 4px;
    padding: 0 8px;
    min-height: 24px;
    max-height: 24px;
}
QLineEdit#DataInput:focus { border-color: #b08a3c; }

QComboBox#DataCombo {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
    background: #ebe9e3;
    border: 1px solid #d8d4cb;
    border-radius: 4px;
    padding: 0 8px;
    min-height: 24px;
    max-height: 24px;
}
QComboBox#DataCombo:focus { border-color: #b08a3c; }
QComboBox#DataCombo::drop-down { border: none; }

/* ── Standard button ─────────────────────────────────────────────────────── */
QPushButton#StdBtn {
    font-size: 12px;
    font-weight: 500;
    color: #1c1b19;
    background: #fffefb;
    border: 1px solid #b9b3a6;
    border-radius: 4px;
    padding: 0 10px;
    min-height: 26px;
    max-height: 26px;
}
QPushButton#StdBtn:hover { background: #ebe9e3; }
QPushButton#StdBtn:pressed { background: #e1ded6; }

QPushButton#PrimaryBtn {
    font-size: 12px;
    font-weight: 500;
    color: #f5f4f1;
    background: #1c1b19;
    border: 1px solid #1c1b19;
    border-radius: 4px;
    padding: 0 12px;
    min-height: 28px;
    max-height: 28px;
}
QPushButton#PrimaryBtn:hover { background: #2a2824; }
QPushButton#PrimaryBtn:pressed { background: #34322d; }

/* ── Status bar ─────────────────────────────────────────────────────────── */
QWidget#StatusBar {
    background: rgba(255,254,251,220);
    border-top: 1px solid #d8d4cb;
    min-height: 24px;
    max-height: 24px;
}
QLabel#StatusText {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 10px;
    color: #7a766e;
}
QLabel#StatusMono {
    font-family: "JetBrains Mono", "Consolas";
    font-size: 10px;
    color: #4a4742;
}

/* ── Separators ──────────────────────────────────────────────────────────── */
QFrame[frameShape="4"],  /* HLine */
QFrame[frameShape="5"] { /* VLine */
    color: #e6e2d9;
}

/* ── Table widget (run history) ──────────────────────────────────────────── */
QTableWidget {
    background: #fffefb;
    alternate-background-color: #f5f4f1;
    gridline-color: #e6e2d9;
    border: none;
    font-family: "JetBrains Mono", "Consolas";
    font-size: 11px;
    color: #1c1b19;
    selection-background-color: #eee9d9;
    selection-color: #1c1b19;
}
QTableWidget::item { padding: 4px 12px; }
QHeaderView::section {
    background: #ebe9e3;
    font-family: "Inter Tight", "Segoe UI";
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #7a766e;
    text-transform: uppercase;
    border: none;
    border-bottom: 1px solid #d8d4cb;
    padding: 6px 12px;
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #d8d4cb;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #b9b3a6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #d8d4cb;
    border-radius: 3px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #b9b3a6; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Navigation sidebar items ────────────────────────────────────────────── */
QPushButton#NavBtn {
    font-size: 12px;
    font-weight: 500;
    color: #4a4742;
    background: transparent;
    border: none;
    border-radius: 4px;
    text-align: left;
    padding: 0 12px;
    min-height: 32px;
    max-height: 32px;
}
QPushButton#NavBtn:hover { background: #ebe9e3; color: #1c1b19; }
QPushButton#NavBtn:checked {
    background: #eee9d9;
    color: #1c1b19;
    font-weight: 600;
}

/* ── Divider label (sidebar section) ────────────────────────────────────── */
QLabel#SectionLabel {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #a8a39a;
    padding: 0 12px;
}

)QSS");
}

inline QString darkSheet()
{
    return QStringLiteral(R"QSS(
QMainWindow, QWidget#AppRoot { background: #1c1b19; }
QWidget#Toolbar { background: rgba(28,27,25,220); border-bottom: 1px solid #34322d; }
QLabel#Wordmark { color: #ece9e2; }
QLabel#ToolbarStatus { background: #25231f; border-color: #34322d; color: #b8b3a8; }
QWidget#Sidebar, QWidget#Inspector { background: rgba(28,27,25,220); border-color: #34322d; }
QFrame#ContentPanel { background: #25231f; border-color: #34322d; }
QLabel#PanelEyebrow { color: #8a857b; }
QLabel#PanelHeading, QLabel#PanelTitle { color: #ece9e2; }
QLabel#LayerName, QLabel#KVVal { color: #ece9e2; }
QLabel#KVKey, QLabel#StatusText { color: #8a857b; }
QLabel#StatusMono { color: #b8b3a8; }
QLabel#Readout { color: #ece9e2; }
QLineEdit#SearchInput, QLineEdit#DataInput, QComboBox#DataCombo {
    background: #181715; border-color: #34322d; color: #ece9e2;
}
QTableWidget { background: #25231f; alternate-background-color: #1c1b19; color: #ece9e2; gridline-color: #2a2824; selection-background-color: #34322d; }
QHeaderView::section { background: #181715; color: #8a857b; border-color: #34322d; }
QWidget#StatusBar { background: rgba(28,27,25,220); border-color: #34322d; }
QPushButton#NavBtn { color: #b8b3a8; }
QPushButton#NavBtn:hover { background: #34322d; color: #ece9e2; }
QPushButton#NavBtn:checked { background: #34322d; color: #ece9e2; }
QPushButton#IconBtn:hover { background: #34322d; color: #ece9e2; }
QPushButton#IconBtn:checked { background: #34322d; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #34322d; }
QLabel#SectionLabel { color: #5a564f; }
)QSS");
}

} // namespace Theme
