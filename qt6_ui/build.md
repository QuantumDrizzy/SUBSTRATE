# SUBSTRATE Qt6 UI — Build Instructions

## Prerequisites

### Windows (MSVC + Qt6)

1. Install **Qt 6.7+** via the Qt online installer: https://www.qt.io/download
   - Select component: `Qt 6.7.x → MSVC 2022 64-bit`
2. Install **Visual Studio 2022** (C++ workload) or **Build Tools for VS 2022**
3. Install **CMake 3.22+** (bundled with VS or standalone)

```powershell
# Configure (from qt6_ui/ directory)
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH="C:/Qt/6.7.0/msvc2022_64"

# Build
cmake --build build --config Release

# Run
.\build\Release\substrate.exe
```

If Qt is on PATH (e.g. added Qt's bin/ to System PATH), CMake finds it automatically.

---

### Windows (MinGW + Qt6)

```powershell
cmake -B build -G "MinGW Makefiles" `
      -DCMAKE_PREFIX_PATH="C:/Qt/6.7.0/mingw_64"

cmake --build build --config Release
```

---

### Arch Linux

```bash
# Install dependencies
sudo pacman -S qt6-base cmake ninja gcc

# Configure & build
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build

# Run
./build/substrate
```

Optional: install fonts for the full design aesthetic:
```bash
# Inter Tight (UI font)
yay -S ttf-inter          # or download from https://rsms.me/inter/

# JetBrains Mono (data/mono font)
sudo pacman -S ttf-jetbrains-mono
```

---

### Ubuntu / Debian

```bash
sudo apt install qt6-base-dev cmake ninja-build build-essential

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/substrate
```

---

## Font fallbacks

If Inter Tight / JetBrains Mono are not installed, the app falls back to:
- **UI:** Segoe UI (Windows) → system-ui
- **Mono:** Consolas (Windows) → monospace

The visual output matches the design closely either way — metric differences
are < 1px on standard DPI.

---

## Dark mode

The app detects the OS color scheme automatically via `QStyleHints::colorScheme()`
(Qt 6.5+). On Windows 11, toggling Settings → Personalisation → Colors → Dark
will switch the palette without restarting.

---

## Project structure

```
qt6_ui/
├── CMakeLists.txt
└── src/
    ├── main.cpp                 # Entry point, QApplication setup
    ├── MainWindow.h/.cpp        # Top-level window, toolbar, body, status bar
    ├── theme/
    │   └── StyleSheet.h         # All QSS tokens (1:1 with design system CSS)
    ├── panels/
    │   ├── DashboardPanel.h/.cpp   # 2×3 physics layer card grid
    │   └── LayerDetailPanel.h/.cpp # Right inspector + solver config
    └── widgets/
        ├── ScoreBar.h/.cpp      # Score bar row (custom-painted)
        └── GlassCard.h/.cpp     # Glassmorphism card container
```

## Design system reference

All visual tokens are derived from `substrate-design-system/project/colors_and_type.css`:
- Background: `#f5f4f1`
- Surface: `#fffefb`
- Glass: `rgba(255,254,251,0.90)` → QColor alpha 220/255
- Score thresholds: HIGH ≥ 0.7, MED ≥ 0.4, LOW < 0.4
- Fonts: Inter Tight (UI), JetBrains Mono (data)
- Radii: 4px controls, 6px panels, 999px pills
