#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <array>
#include "panels/DashboardPanel.h"

class DashboardPanel;
class LayerDetailPanel;
class QPushButton;

// MainWindow — top-level SUBSTRATE desktop window.
// Chrome: toolbar (40px) | body (sidebar + main + inspector) | status bar (24px).
// Toolbar: wordmark · run/stop/pause · view toggle · run status · search/settings
// Sidebar: nav (Dashboard / Run Analysis / History / Settings) + layer list
// Main: stacked (Dashboard grid | Run history table)
// Right: LayerDetailPanel (inspector + config)
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

protected:
    void closeEvent(QCloseEvent*) override;

private slots:
    void onRunToggle();
    void onLayerClicked(int index);
    void onNavClicked(int index);
    void tickSimulation();

private:
    void buildToolbar();
    void buildBody();
    void buildStatusBar();
    void seedData();
    void refreshStatus();
    void applyTheme();

    // ── Toolbar widgets
    QPushButton* m_runBtn   = nullptr;
    QPushButton* m_stopBtn  = nullptr;
    QPushButton* m_pauseBtn = nullptr;
    QLabel*      m_runStatus = nullptr;

    // ── Nav
    std::array<QPushButton*, 4> m_navBtns{};

    // ── Content panels
    DashboardPanel*   m_dashboard  = nullptr;
    LayerDetailPanel* m_inspector  = nullptr;
    QWidget*          m_mainStack  = nullptr;

    // ── Status bar
    QLabel* m_sbLeft  = nullptr;
    QLabel* m_sbRight = nullptr;

    // ── Simulation state
    bool   m_running  = false;
    int    m_runId    = 481;
    int    m_iter     = 0;
    int    m_maxIter  = 1000;
    double m_residual = 1.0;
    QTimer* m_ticker  = nullptr;

    std::array<LayerData, 6> m_layers{};
    int m_selectedLayer = -1;
};
