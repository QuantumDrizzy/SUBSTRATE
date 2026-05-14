#include "LayerDetailPanel.h"
#include "widgets/ScoreBar.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QFrame>
#include <QPainter>
#include <QPainterPath>
#include <QScrollArea>

// ── Helpers ───────────────────────────────────────────────────────────────────

static QLabel* eyebrow(const QString& text, QWidget* parent = nullptr)
{
    auto* l = new QLabel(text.toUpper(), parent);
    l->setObjectName("PanelEyebrow");
    return l;
}

static QLabel* kvKey(const QString& text, QWidget* parent = nullptr)
{
    auto* l = new QLabel(text, parent);
    l->setObjectName("KVKey");
    return l;
}

static QLabel* kvVal(const QString& text, QWidget* parent = nullptr)
{
    auto* l = new QLabel(text, parent);
    l->setObjectName("KVVal");
    return l;
}

static QFrame* hline(QWidget* parent = nullptr)
{
    auto* f = new QFrame(parent);
    f->setFrameShape(QFrame::HLine);
    f->setFixedHeight(1);
    return f;
}

static QLabel* sectionLabel(const QString& text, QWidget* parent = nullptr)
{
    auto* l = new QLabel(text, parent);
    l->setObjectName("SectionLabel");
    return l;
}

// ── LayerDetailPanel ──────────────────────────────────────────────────────────

LayerDetailPanel::LayerDetailPanel(QWidget* parent) : QWidget(parent)
{
    setObjectName("Inspector");
    setMinimumWidth(240);
    setMaximumWidth(340);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // Panel header
    auto* header = new QWidget;
    header->setFixedHeight(38);
    auto* hlay = new QHBoxLayout(header);
    hlay->setContentsMargins(14, 0, 12, 0);
    hlay->addWidget(eyebrow("Inspector"));
    hlay->addStretch();
    root->addWidget(header);
    root->addWidget(hline());

    // Scrollable content
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* croot   = new QVBoxLayout(content);
    croot->setContentsMargins(0, 0, 0, 0);
    croot->setSpacing(0);

    // Empty state
    m_emptyMsg = new QLabel("Select a layer to inspect.");
    m_emptyMsg->setObjectName("KVKey");
    m_emptyMsg->setAlignment(Qt::AlignCenter);
    m_emptyMsg->setContentsMargins(16, 20, 16, 20);
    croot->addWidget(m_emptyMsg);

    // ── Layer detail section ──────────────────────────────────────────────
    m_detailSection = new QWidget;
    auto* dlay = new QVBoxLayout(m_detailSection);
    dlay->setContentsMargins(14, 12, 14, 12);
    dlay->setSpacing(0);

    // id row + dot
    auto* idRow = new QHBoxLayout;
    idRow->setSpacing(8);
    auto* dot = new QLabel("●");
    dot->setObjectName("KVKey");
    dot->setFixedWidth(14);
    m_idLabel = kvVal("—");
    idRow->addWidget(dot);
    idRow->addWidget(m_idLabel);
    idRow->addStretch();
    dlay->addLayout(idRow);
    dlay->addSpacing(6);

    // Score readout + pill
    auto* readRow = new QHBoxLayout;
    readRow->setSpacing(8);
    m_readout = new QLabel("—");
    m_readout->setObjectName("Readout");
    m_pill = new QLabel("IDLE");
    m_pill->setObjectName("PillIdle");
    readRow->addWidget(m_readout);
    readRow->addStretch();
    readRow->addWidget(m_pill);
    dlay->addLayout(readRow);
    dlay->addSpacing(10);

    // Score bar
    m_bar = new ScoreBar;
    m_bar->setLayer("—", -1, true);
    dlay->addWidget(m_bar);
    dlay->addSpacing(10);

    // KV table
    auto* kvGrid = new QGridLayout;
    kvGrid->setSpacing(0);
    kvGrid->setContentsMargins(0, 0, 0, 0);
    kvGrid->setColumnMinimumWidth(0, 70);

    auto addKV = [&](int row, const QString& key, QLabel*& valOut) {
        auto* fr = new QFrame;
        fr->setFrameShape(QFrame::HLine);
        fr->setFixedHeight(1);
        valOut = kvVal("—");
        kvGrid->addWidget(kvKey(key), row, 0);
        kvGrid->addWidget(valOut,     row, 1);
        if (row > 0) {
            // thin separator above each row
        }
    };

    addKV(0, "type",    m_kvType);
    addKV(1, "μ",       m_kvMu);
    addKV(2, "σ",       m_kvSigma);
    addKV(3, "visible", m_kvVisible);
    addKV(4, "locked",  m_kvLocked);
    dlay->addLayout(kvGrid);

    croot->addWidget(m_detailSection);
    m_detailSection->hide();

    croot->addWidget(hline());

    // ── Solver config section ─────────────────────────────────────────────
    auto* cfgSection = new QWidget;
    auto* cfgLay = new QVBoxLayout(cfgSection);
    cfgLay->setContentsMargins(14, 12, 14, 12);
    cfgLay->setSpacing(6);

    cfgLay->addWidget(sectionLabel("Solver Configuration"));
    cfgLay->addSpacing(4);

    auto addInput = [&](const QString& label, QWidget* input) {
        auto* row = new QHBoxLayout;
        row->setSpacing(10);
        auto* lbl = kvKey(label);
        lbl->setFixedWidth(100);
        row->addWidget(lbl);
        row->addWidget(input, 1);
        cfgLay->addLayout(row);
    };

    m_algCombo = new QComboBox;
    m_algCombo->setObjectName("DataCombo");
    m_algCombo->addItem("Levenberg–Marquardt");
    m_algCombo->addItem("Gauss–Newton");
    m_algCombo->addItem("Trust Region");
    addInput("Algorithm", m_algCombo);

    m_tolEdit = new QLineEdit("1.0e−6");
    m_tolEdit->setObjectName("DataInput");
    addInput("Tolerance", m_tolEdit);

    m_iterEdit = new QLineEdit("1000");
    m_iterEdit->setObjectName("DataInput");
    addInput("Max iter", m_iterEdit);

    m_dampEdit = new QLineEdit("0.001");
    m_dampEdit->setObjectName("DataInput");
    addInput("Damping λ₀", m_dampEdit);

    croot->addWidget(cfgSection);
    croot->addStretch();

    scroll->setWidget(content);
    root->addWidget(scroll, 1);
}

void LayerDetailPanel::showLayer(const LayerData& data)
{
    m_emptyMsg->hide();
    m_detailSection->show();

    m_idLabel->setText(data.id);

    const bool idle = data.score < 0;
    const QString scoreStr = idle ? "—" : QString::number(data.score, 'f', 3);
    m_readout->setText(scoreStr);

    // Pill state
    QString pillId, pillText;
    if (idle)              { pillId = "PillIdle"; pillText = "IDLE"; }
    else if (data.score >= 0.7) { pillId = "PillHigh"; pillText = "HIGH"; }
    else if (data.score >= 0.4) { pillId = "PillMed";  pillText = "MED";  }
    else                        { pillId = "PillLow";  pillText = "LOW";  }
    m_pill->setObjectName(pillId);
    m_pill->setText(pillText);
    m_pill->style()->unpolish(m_pill);
    m_pill->style()->polish(m_pill);

    // Score bar
    m_bar->setLayer(data.id, data.score, idle);

    // KV rows
    m_kvType   ->setText(data.type);
    m_kvMu     ->setText(idle ? "—" : QString::number(data.mu,    'f', 4));
    m_kvSigma  ->setText(idle ? "—" : QString::number(data.sigma, 'f', 5));
    m_kvVisible->setText("true");
    m_kvLocked ->setText("false");
}

void LayerDetailPanel::clearLayer()
{
    m_detailSection->hide();
    m_emptyMsg->show();
}
