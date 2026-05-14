#include "DashboardPanel.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFontInfo>
#include <QFontMetrics>

// ── LayerCard — internal implementation ──────────────────────────────────────
// Defined in the .cpp so DashboardPanel.h stays clean.

class LayerCard : public QWidget
{
    Q_OBJECT
public:
    explicit LayerCard(int index, QWidget* parent = nullptr)
        : QWidget(parent), m_index(index)
    {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(240, 168);
        setCursor(Qt::PointingHandCursor);
    }

    void setData(const LayerData& d) { m_data = d; update(); }

signals:
    void clicked(int index);

protected:
    void mousePressEvent(QMouseEvent*) override { emit clicked(m_index); }

    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        const qreal  radius = 6.0;

        // Card background
        QPainterPath cardPath;
        cardPath.addRoundedRect(r, radius, radius);
        p.setPen(Qt::NoPen);
        p.fillPath(cardPath, QColor(255, 254, 251, 235));

        // Border
        p.setPen(QPen(QColor(0xd8, 0xd4, 0xcb), 1.0));
        p.drawPath(cardPath);

        // Top inner highlight
        p.setPen(QPen(QColor(255, 255, 255, 80), 1.0));
        const qreal hx1 = r.left() + radius + 1;
        const qreal hx2 = r.right() - radius - 1;
        p.drawLine(QPointF(hx1, r.top() + 1.0), QPointF(hx2, r.top() + 1.0));

        // ── Interior ──────────────────────────────────────────────────────
        const int px = 16, py = 13;
        int y = py;
        const int W = r.width() - px * 2;

        auto [pillBg, pillFg, pillLabel, scoreColor] = stateColors();

        // Eyebrow (layer id, 10px bold tracked uppercase)
        QFont eyeF = uiFont(10, QFont::DemiBold);
        p.setFont(eyeF);
        p.setPen(QColor(0x7a, 0x76, 0x6e));
        p.drawText(QRectF(px, y, W, 14), Qt::AlignLeft | Qt::AlignVCenter,
                   m_data.id.toUpper());
        y += 16;

        // Layer name (13px medium)
        QFont nameF = uiFont(12, QFont::Medium);
        p.setFont(nameF);
        p.setPen(QColor(0x4a, 0x47, 0x42));
        p.drawText(QRectF(px, y, W - 58, 18), Qt::AlignLeft | Qt::AlignVCenter,
                   QFontMetrics(nameF).elidedText(m_data.name, Qt::ElideRight, W - 58));

        // Pill (top-right, aligned with name row)
        drawPill(p, px + W, y + 1, pillBg, pillFg, pillLabel);

        y += 22;

        // Big score readout (28px mono)
        QFont readF = monoFont(26, QFont::Medium);
        p.setFont(readF);
        p.setPen(scoreColor);
        const QString scoreStr = (m_data.score < 0) ? "—"
                                : QString::number(m_data.score, 'f', 3);
        p.drawText(QRectF(px, y, W, 34), Qt::AlignLeft | Qt::AlignVCenter, scoreStr);
        y += 38;

        // Score bar
        const int barH = 8;
        drawBar(p, px, y + 2, W, barH, m_data.score, scoreColor);
        y += barH + 14;

        // Divider
        p.setPen(QPen(QColor(0xe6, 0xe2, 0xd9), 1));
        p.drawLine(QPointF(px, y), QPointF(px + W, y));
        y += 9;

        // μ / σ metadata rows
        drawKV(p, px, y,     W, "μ",  (m_data.score < 0) ? "—" : QString::number(m_data.mu,    'f', 4));
        drawKV(p, px, y + 20, W, "σ", (m_data.score < 0) ? "—" : QString::number(m_data.sigma, 'f', 5));

        // Running dot (bottom-right)
        if (m_data.running) {
            p.setPen(Qt::NoPen);
            QPainterPath dot;
            dot.addEllipse(QPointF(r.right() - 10, r.bottom() - 10), 4, 4);
            p.fillPath(dot, QColor(0xa0, 0x70, 0x20));
        }
    }

private:
    int       m_index;
    LayerData m_data{};

    struct StateColors {
        QColor pillBg, pillFg;
        QString label;
        QColor score;
    };

    StateColors stateColors() const {
        if (m_data.score < 0)
            return { QColor(0xe8,0xe5,0xdd), QColor(0x8a,0x85,0x7b), "IDLE", QColor(0x8a,0x85,0x7b) };
        if (m_data.score >= 0.7)
            return { QColor(0xe3,0xee,0xe5), QColor(0x2f,0x7a,0x4d), "HIGH", QColor(0x2f,0x7a,0x4d) };
        if (m_data.score >= 0.4)
            return { QColor(0xf3,0xea,0xd6), QColor(0xa0,0x70,0x20), "MED",  QColor(0xa0,0x70,0x20) };
        return { QColor(0xf0,0xdf,0xdb), QColor(0xa8,0x39,0x2f), "LOW",  QColor(0xa8,0x39,0x2f) };
    }

    static QFont uiFont(int px, QFont::Weight w = QFont::Normal) {
        QFont f("Inter Tight");
        f.setPixelSize(px);
        f.setWeight(w);
        if (!QFontInfo(f).family().contains("Inter", Qt::CaseInsensitive))
            f.setFamily("Segoe UI");
        return f;
    }

    static QFont monoFont(int px, QFont::Weight w = QFont::Normal) {
        QFont f("JetBrains Mono");
        f.setPixelSize(px);
        f.setWeight(w);
        if (!QFontInfo(f).fixedPitch())
            f.setFamily("Consolas");
        return f;
    }

    static void drawPill(QPainter& p, int rightEdge, int y,
                         QColor bg, QColor fg, const QString& label)
    {
        QFont f("Inter Tight");
        f.setPixelSize(10);
        f.setWeight(QFont::DemiBold);
        if (!QFontInfo(f).family().contains("Inter", Qt::CaseInsensitive))
            f.setFamily("Segoe UI");
        p.setFont(f);
        QFontMetrics fm(f);
        const int ph = 18, ppx = 8;
        const int pw = fm.horizontalAdvance(label) + ppx * 2;
        const int px = rightEdge - pw;
        QPainterPath pp;
        pp.addRoundedRect(QRectF(px, y, pw, ph), 999, 999);
        p.setPen(Qt::NoPen);
        p.fillPath(pp, bg);
        p.setPen(fg);
        p.drawText(QRect(px, y, pw, ph), Qt::AlignCenter, label);
    }

    static void drawBar(QPainter& p, int x, int y, int w, int h,
                        double score, QColor fillColor)
    {
        QRect tr(x, y, w, h);
        QPainterPath tp;
        tp.addRoundedRect(tr, 2, 2);
        p.setPen(Qt::NoPen);
        p.fillPath(tp, QColor(0xeb, 0xe9, 0xe3));
        p.setPen(QPen(QColor(0, 0, 0, 10), 1));
        p.drawPath(tp);

        if (score >= 0) {
            const double fill = qBound(0.0, score, 1.0);
            QRect fr(tr.x(), tr.y(), qRound(tr.width() * fill), tr.height());
            if (fr.width() > 0) {
                QPainterPath fp;
                fp.addRoundedRect(fr, 2, 2);
                p.setPen(Qt::NoPen);
                p.fillPath(fp, fillColor);
            }
        }
        // Threshold markers at 0.4 and 0.7
        p.setPen(QPen(QColor(0xb9, 0xb3, 0xa6, 110), 1));
        for (double t : {0.4, 0.7}) {
            int mx = tr.x() + qRound(tr.width() * t);
            p.drawLine(mx, tr.top(), mx, tr.bottom());
        }
    }

    static void drawKV(QPainter& p, int x, int y, int w,
                       const QString& key, const QString& val)
    {
        QFont kf("Inter Tight");
        kf.setPixelSize(11);
        if (!QFontInfo(kf).family().contains("Inter", Qt::CaseInsensitive))
            kf.setFamily("Segoe UI");
        p.setFont(kf);
        p.setPen(QColor(0x7a, 0x76, 0x6e));
        p.drawText(QRect(x, y, 28, 16), Qt::AlignLeft | Qt::AlignVCenter, key);

        QFont vf("JetBrains Mono");
        vf.setPixelSize(11);
        if (!QFontInfo(vf).fixedPitch()) vf.setFamily("Consolas");
        p.setFont(vf);
        p.setPen(QColor(0x1c, 0x1b, 0x19));
        p.drawText(QRect(x + 28, y, w - 28, 16), Qt::AlignLeft | Qt::AlignVCenter, val);
    }
};

// ── DashboardPanel ───────────────────────────────────────────────────────────

DashboardPanel::DashboardPanel(QWidget* parent) : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // Header row
    auto* headerW = new QWidget;
    headerW->setFixedHeight(44);
    auto* hlay = new QHBoxLayout(headerW);
    hlay->setContentsMargins(4, 0, 4, 0);
    hlay->setSpacing(10);

    auto* eyebrow = new QLabel("OVERVIEW");
    eyebrow->setObjectName("PanelEyebrow");
    auto* title = new QLabel("Physics Layer Dashboard");
    title->setObjectName("PanelHeading");
    hlay->addWidget(eyebrow);
    hlay->addWidget(title);
    hlay->addStretch();
    root->addWidget(headerW);

    // 2×3 grid (2 rows, 3 cols)
    auto* gridW  = new QWidget;
    auto* grid   = new QGridLayout(gridW);
    grid->setSpacing(8);
    grid->setContentsMargins(4, 0, 4, 4);

    for (int i = 0; i < 6; ++i) {
        auto* card = new LayerCard(i, this);
        connect(card, &LayerCard::clicked, this, &DashboardPanel::layerClicked);
        m_cards[i] = card;
        grid->addWidget(card, i / 3, i % 3);
    }
    for (int c = 0; c < 3; ++c) grid->setColumnStretch(c, 1);
    for (int r = 0; r < 2; ++r) grid->setRowStretch(r, 1);

    root->addWidget(gridW, 1);
}

void DashboardPanel::updateLayer(int index, const LayerData& data)
{
    if (index >= 0 && index < 6)
        m_cards[index]->setData(data);
}

#include "DashboardPanel.moc"
