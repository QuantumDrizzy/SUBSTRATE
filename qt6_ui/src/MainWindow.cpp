#include "MainWindow.h"
#include "panels/DashboardPanel.h"
#include "panels/LayerDetailPanel.h"
#include "widgets/ScoreBar.h"
#include "theme/StyleSheet.h"

#include <QApplication>
#include <QStyleHints>
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFrame>
#include <QFont>
#include <QFontDatabase>
#include <QTimer>
#include <QStackedWidget>
#include <QScrollArea>
#include <QTableWidget>
#include <QHeaderView>
#include <QCloseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QSizePolicy>
#include <cmath>

// ── SVG-path icon button helper ───────────────────────────────────────────────
// Renders a 16×16 SVG path as button icon text (placeholder — real icons via
// QIcon in production). Here we use Unicode symbols as stand-ins.

static QPushButton* iconBtn(const QString& symbol, const QString& tooltip,
                            bool checkable = false, QWidget* parent = nullptr)
{
    auto* btn = new QPushButton(symbol, parent);
    btn->setObjectName("IconBtn");
    btn->setToolTip(tooltip);
    btn->setCheckable(checkable);
    btn->setFixedSize(28, 28);
    return btn;
}

static QFrame* vSep(QWidget* parent = nullptr)
{
    auto* f = new QFrame(parent);
    f->setFrameShape(QFrame::VLine);
    f->setFixedWidth(1);
    f->setFixedHeight(20);
    return f;
}

static QFrame* hLine(QWidget* parent = nullptr)
{
    auto* f = new QFrame(parent);
    f->setFrameShape(QFrame::HLine);
    f->setFixedHeight(1);
    return f;
}

// ── MainWindow ────────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    setWindowTitle("SUBSTRATE — Unified Field Analysis System");
    setMinimumSize(1280, 720);
    resize(1440, 900);

    applyTheme();
    seedData();

    // Central widget
    auto* root = new QWidget;
    root->setObjectName("AppRoot");
    setCentralWidget(root);

    auto* vroot = new QVBoxLayout(root);
    vroot->setContentsMargins(0, 0, 0, 0);
    vroot->setSpacing(0);

    buildToolbar();
    vroot->addWidget(findChild<QWidget*>("Toolbar"));

    buildBody();
    vroot->addWidget(findChild<QWidget*>("Body"), 1);

    buildStatusBar();
    vroot->addWidget(findChild<QWidget*>("StatusBar"));

    // Simulation ticker
    m_ticker = new QTimer(this);
    m_ticker->setInterval(80);
    connect(m_ticker, &QTimer::timeout, this, &MainWindow::tickSimulation);
}

void MainWindow::applyTheme()
{
    QString sheet = Theme::lightSheet();

    // Auto-detect dark mode (Qt 6.5+)
    const auto* hints = QApplication::styleHints();
    if (hints && hints->colorScheme() == Qt::ColorScheme::Dark)
        sheet += Theme::darkSheet();

    qApp->setStyleSheet(sheet);
}

// ── Toolbar ───────────────────────────────────────────────────────────────────

void MainWindow::buildToolbar()
{
    auto* tb = new QWidget(centralWidget());
    tb->setObjectName("Toolbar");
    tb->setFixedHeight(40);

    auto* lay = new QHBoxLayout(tb);
    lay->setContentsMargins(12, 0, 12, 0);
    lay->setSpacing(2);

    // Wordmark
    auto* wm = new QLabel("SUBSTRATE");
    wm->setObjectName("Wordmark");
    lay->addWidget(wm);

    lay->addSpacing(4);
    lay->addWidget(vSep(tb));
    lay->addSpacing(4);

    // Run controls
    m_runBtn   = iconBtn("▶", "Run",   true, tb);
    m_stopBtn  = iconBtn("■", "Stop",  false, tb);
    m_pauseBtn = iconBtn("⏸", "Pause", true, tb);
    lay->addWidget(m_runBtn);
    lay->addWidget(m_stopBtn);
    lay->addWidget(m_pauseBtn);

    connect(m_runBtn,   &QPushButton::clicked, this, &MainWindow::onRunToggle);
    connect(m_stopBtn,  &QPushButton::clicked, this, [this]{ onRunToggle(); });
    connect(m_pauseBtn, &QPushButton::clicked, this, [this]{
        if (m_running) { m_ticker->stop(); m_pauseBtn->setChecked(true); }
        else           { m_ticker->start(); m_pauseBtn->setChecked(false); }
    });

    lay->addSpacing(4);
    lay->addWidget(vSep(tb));
    lay->addSpacing(4);

    // View toggle (Dashboard / Runs)
    auto* dashBtn = iconBtn("⊞", "Dashboard", true, tb);
    dashBtn->setObjectName("IconBtn");
    dashBtn->setChecked(true);
    auto* runsBtn = iconBtn("≡", "Run History", true, tb);
    runsBtn->setObjectName("IconBtn");
    lay->addWidget(dashBtn);
    lay->addWidget(runsBtn);
    connect(dashBtn, &QPushButton::clicked, this, [this, dashBtn, runsBtn]{
        dashBtn->setChecked(true); runsBtn->setChecked(false);
        onNavClicked(0);
    });
    connect(runsBtn, &QPushButton::clicked, this, [this, dashBtn, runsBtn]{
        runsBtn->setChecked(true); dashBtn->setChecked(false);
        onNavClicked(2);
    });

    lay->addStretch();

    // Run status display
    m_runStatus = new QLabel;
    m_runStatus->setObjectName("ToolbarStatus");
    m_runStatus->setFixedHeight(26);
    m_runStatus->setContentsMargins(10, 0, 10, 0);
    refreshStatus();
    lay->addWidget(m_runStatus);

    lay->addSpacing(4);
    lay->addWidget(vSep(tb));
    lay->addSpacing(4);

    lay->addWidget(iconBtn("⌕", "Search",   false, tb));
    lay->addWidget(iconBtn("⚙", "Settings", false, tb));
}

// ── Body ──────────────────────────────────────────────────────────────────────

void MainWindow::buildBody()
{
    auto* body = new QWidget(centralWidget());
    body->setObjectName("Body");

    auto* lay = new QHBoxLayout(body);
    lay->setContentsMargins(8, 8, 8, 8);
    lay->setSpacing(8);

    // Left sidebar (nav + layer list)
    auto* sidebar = new QWidget;
    sidebar->setObjectName("Sidebar");
    sidebar->setFixedWidth(220);
    {
        auto* sl = new QVBoxLayout(sidebar);
        sl->setContentsMargins(0, 0, 0, 0);
        sl->setSpacing(0);

        auto* secLbl = new QLabel("LAYERS");
        secLbl->setObjectName("SectionLabel");
        secLbl->setContentsMargins(14, 8, 14, 4);
        sl->addWidget(secLbl);

        // Nav buttons
        const QStringList labels = { "Dashboard", "Run Analysis", "History", "Settings" };
        sl->addSpacing(4);
        for (int i = 0; i < 4; ++i) {
            m_navBtns[i] = new QPushButton(labels[i]);
            m_navBtns[i]->setObjectName("NavBtn");
            m_navBtns[i]->setCheckable(true);
            m_navBtns[i]->setChecked(i == 0);
            m_navBtns[i]->setFlat(true);
            sl->addWidget(m_navBtns[i]);
            connect(m_navBtns[i], &QPushButton::clicked, this, [this, i]{ onNavClicked(i); });
        }
        sl->addSpacing(8);
        sl->addWidget(hLine(sidebar));
        sl->addSpacing(6);

        auto* layersSec = new QLabel("LAYERS");
        layersSec->setObjectName("SectionLabel");
        layersSec->setContentsMargins(14, 0, 14, 4);
        sl->addWidget(layersSec);

        // Score bar rows for each layer
        auto* scroll = new QScrollArea;
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* listW = new QWidget;
        auto* listL = new QVBoxLayout(listW);
        listL->setContentsMargins(4, 0, 4, 4);
        listL->setSpacing(1);

        for (int i = 0; i < 6; ++i) {
            const auto& ld = m_layers[i];
            auto* bar = new ScoreBar;
            bar->setLayer(ld.name.left(14), ld.score, ld.score < 0);
            listL->addWidget(bar);
        }
        listL->addStretch();
        scroll->setWidget(listW);
        sl->addWidget(scroll, 1);
    }
    lay->addWidget(sidebar);

    // Main content (stacked)
    auto* stack = new QStackedWidget;
    stack->setObjectName("MainStack");
    m_mainStack = stack;

    // Page 0: Dashboard grid
    m_dashboard = new DashboardPanel;
    for (int i = 0; i < 6; ++i)
        m_dashboard->updateLayer(i, m_layers[i]);
    connect(m_dashboard, &DashboardPanel::layerClicked, this, &MainWindow::onLayerClicked);
    stack->addWidget(m_dashboard);

    // Page 1: placeholder (Run Analysis)
    auto* runPage = new QWidget;
    {
        auto* pl = new QVBoxLayout(runPage);
        pl->addStretch();
        auto* lbl = new QLabel("Run Analysis — configure and launch a new computation.");
        lbl->setObjectName("KVKey");
        lbl->setAlignment(Qt::AlignCenter);
        pl->addWidget(lbl);
        pl->addStretch();
    }
    stack->addWidget(runPage);

    // Page 2: Run history table
    auto* histPage = new QWidget;
    {
        auto* hl = new QVBoxLayout(histPage);
        hl->setContentsMargins(0, 0, 0, 0);
        hl->setSpacing(0);

        auto* header = new QWidget;
        header->setFixedHeight(44);
        auto* hh = new QHBoxLayout(header);
        hh->setContentsMargins(14, 0, 12, 0);
        auto* eyebrow = new QLabel("HISTORY");
        eyebrow->setObjectName("PanelEyebrow");
        auto* htitle = new QLabel("Run History");
        htitle->setObjectName("PanelHeading");
        hh->addWidget(eyebrow);
        hh->addSpacing(10);
        hh->addWidget(htitle);
        hh->addStretch();
        hl->addWidget(header);
        hl->addWidget(hLine(histPage));

        auto* tbl = new QTableWidget(7, 6);
        tbl->setHorizontalHeaderLabels({"Run", "Started", "Elapsed", "Iter", "Residual", "State"});
        tbl->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        tbl->verticalHeader()->hide();
        tbl->setShowGrid(false);
        tbl->setAlternatingRowColors(true);
        tbl->setSelectionBehavior(QAbstractItemView::SelectRows);
        tbl->setEditTriggers(QAbstractItemView::NoEditTriggers);

        const struct RunRow { QString id, at, elapsed, iter, res, state; } rows[] = {
            {"0481","14:27","00:04:17","0247/1000","4.2e−3","RUNNING"},
            {"0480","13:58","00:08:02","1000/1000","9.1e−7","DONE"},
            {"0479","13:42","00:00:34","0089/1000","—","ABORT"},
            {"0478","11:14","00:12:47","1000/1000","1.3e−6","DONE"},
            {"0477","10:52","00:04:08","0512/1000","—","FAILED"},
            {"0476","10:31","00:09:21","1000/1000","8.4e−7","DONE"},
            {"0475","09:48","00:11:02","1000/1000","1.1e−6","DONE"},
        };
        for (int r = 0; r < 7; ++r) {
            tbl->setItem(r, 0, new QTableWidgetItem(rows[r].id));
            tbl->setItem(r, 1, new QTableWidgetItem(rows[r].at));
            tbl->setItem(r, 2, new QTableWidgetItem(rows[r].elapsed));
            tbl->setItem(r, 3, new QTableWidgetItem(rows[r].iter));
            tbl->setItem(r, 4, new QTableWidgetItem(rows[r].res));
            tbl->setItem(r, 5, new QTableWidgetItem(rows[r].state));
            tbl->setRowHeight(r, 28);
        }
        hl->addWidget(tbl, 1);
    }
    stack->addWidget(histPage);

    // Page 3: Settings placeholder
    auto* settingsPage = new QWidget;
    {
        auto* pl = new QVBoxLayout(settingsPage);
        pl->addStretch();
        auto* lbl = new QLabel("Settings");
        lbl->setObjectName("PanelHeading");
        lbl->setAlignment(Qt::AlignCenter);
        pl->addWidget(lbl);
        pl->addStretch();
    }
    stack->addWidget(settingsPage);

    lay->addWidget(stack, 1);

    // Right: inspector
    m_inspector = new LayerDetailPanel;
    lay->addWidget(m_inspector);
}

// ── Status bar ────────────────────────────────────────────────────────────────

void MainWindow::buildStatusBar()
{
    auto* sb = new QWidget(centralWidget());
    sb->setObjectName("StatusBar");
    sb->setFixedHeight(24);

    auto* lay = new QHBoxLayout(sb);
    lay->setContentsMargins(12, 0, 12, 0);
    lay->setSpacing(8);

    m_sbLeft = new QLabel;
    m_sbLeft->setObjectName("StatusText");
    m_sbRight = new QLabel("v0.4.2");
    m_sbRight->setObjectName("StatusText");

    lay->addWidget(m_sbLeft);
    lay->addStretch();
    lay->addWidget(m_sbRight);
    refreshStatus();
}

// ── Data seeding ──────────────────────────────────────────────────────────────

void MainWindow::seedData()
{
    // The 6 SUBSTRATE physics engine layers
    m_layers[0] = { "GEO",   "Geomagnetic",  "RandomForest cycle",      0.847,  0.523, 0.0021, false };
    m_layers[1] = { "QTM",   "Quantum",       "FAD radical pair Lindblad", 0.523,  0.412, 0.0083, false };
    m_layers[2] = { "MGN",   "Magnon",        "Lindblad biosensing",      0.742,  0.601, 0.0034, false };
    m_layers[3] = { "QLAB",  "qLab",          "PEPS tensor network",      0.214, -0.034, 0.0412, false };
    m_layers[4] = { "SOL",   "Solar",         "SC25 sinusoidal model",    0.689,  0.482, 0.0067, false };
    m_layers[5] = { "COSMO", "Cosmo",         "CMB spherical harmonic",   -1.0,   0.0,   0.0,    false };
}

// ── Slots ─────────────────────────────────────────────────────────────────────

void MainWindow::onRunToggle()
{
    m_running = !m_running;
    m_runBtn->setChecked(m_running);
    if (m_running) {
        m_iter = 0;
        m_residual = 1.0;
        ++m_runId;
        m_ticker->start();
    } else {
        m_ticker->stop();
    }
    refreshStatus();
}

void MainWindow::onLayerClicked(int index)
{
    m_selectedLayer = index;
    m_inspector->showLayer(m_layers[index]);
    refreshStatus();
}

void MainWindow::onNavClicked(int index)
{
    for (int i = 0; i < 4; ++i)
        m_navBtns[i]->setChecked(i == index);
    if (auto* stack = findChild<QStackedWidget*>("MainStack"))
        stack->setCurrentIndex(index);
}

void MainWindow::tickSimulation()
{
    ++m_iter;
    m_residual *= (1.0 - 0.003 * (0.8 + 0.2 * (double(m_iter % 17) / 17.0)));

    // Animate layer scores
    for (int i = 0; i < 6; ++i) {
        if (m_layers[i].score < 0) continue;
        const double drift = 0.001 * std::sin(m_iter * 0.1 + i * 1.3);
        m_layers[i].score  = qBound(0.01, m_layers[i].score + drift, 0.999);
        m_layers[i].sigma  = qMax(0.0001, m_layers[i].sigma * (1.0 + 0.002 * std::sin(m_iter * 0.07 + i)));
        m_dashboard->updateLayer(i, m_layers[i]);
    }

    if (m_iter >= m_maxIter) {
        m_ticker->stop();
        m_running = false;
        m_runBtn->setChecked(false);
    }

    refreshStatus();
}

void MainWindow::refreshStatus()
{
    const QString state = m_running ? "RUNNING" : (m_iter > 0 ? "DONE" : "IDLE");
    const QString runStr = QString("RUN %1  ·  iter %2/%3  ·  res %4")
        .arg(m_runId, 4, 10, QChar('0'))
        .arg(m_iter,  4, 10, QChar('0'))
        .arg(m_maxIter)
        .arg(m_residual, 0, 'e', 2);

    if (m_runStatus) m_runStatus->setText(runStr);
    if (m_sbLeft) {
        const QString sel = m_selectedLayer >= 0
            ? QString("selected: %1").arg(m_layers[m_selectedLayer].id)
            : "no selection";
        m_sbLeft->setText(QString("%1  ·  %2  ·  6/6 visible").arg(state, sel));
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    m_ticker->stop();
    event->accept();
}

#include "MainWindow.moc"
