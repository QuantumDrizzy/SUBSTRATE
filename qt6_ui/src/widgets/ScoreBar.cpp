#include "ScoreBar.h"
#include <QPainter>
#include <QPainterPath>
#include <QFontInfo>
#include <QFontMetrics>

ScoreBar::ScoreBar(QWidget* parent) : QWidget(parent)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setFixedHeight(32);
}

void ScoreBar::setLayer(const QString& name, double score, bool idle)
{
    m_name  = name;
    m_score = score;
    m_idle  = idle;
    update();
}

ScoreBar::Colors ScoreBar::colorsForScore() const
{
    if (m_idle || m_score < 0)
        return { QColor(0x8a,0x85,0x7b), QColor(0x8a,0x85,0x7b),
                 QColor(0xe8,0xe5,0xdd), QColor(0x8a,0x85,0x7b), "IDLE" };
    if (m_score >= 0.7)
        return { QColor(0x2f,0x7a,0x4d), QColor(0x2f,0x7a,0x4d),
                 QColor(0xe3,0xee,0xe5), QColor(0x2f,0x7a,0x4d), "HIGH" };
    if (m_score >= 0.4)
        return { QColor(0xa0,0x70,0x20), QColor(0xa0,0x70,0x20),
                 QColor(0xf3,0xea,0xd6), QColor(0xa0,0x70,0x20), "MED" };
    return { QColor(0xa8,0x39,0x2f), QColor(0xa8,0x39,0x2f),
             QColor(0xf0,0xdf,0xdb), QColor(0xa8,0x39,0x2f), "LOW" };
}

void ScoreBar::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const Colors c = colorsForScore();
    const QRect  r = rect();

    // Layout columns (mirrors CSS grid: 110px 1fr 60px 60px, gap 12px)
    const int pad    = 12;
    const int nameW  = 110;
    const int valW   = 52;
    const int pillW  = 52;
    const int gap    = 12;
    int x = pad;

    // ── Layer name ────────────────────────────────────────────────────────
    QFont monoFont("JetBrains Mono");
    monoFont.setPixelSize(11);
    if (!QFontInfo(monoFont).fixedPitch())
        monoFont.setFamily("Consolas");
    monoFont.setWeight(QFont::Normal);
    p.setFont(monoFont);
    p.setPen(QColor(0x4a, 0x47, 0x42));
    QRect nameRect(x, 0, nameW, r.height());
    p.drawText(nameRect, Qt::AlignVCenter | Qt::AlignLeft,
               QFontMetrics(monoFont).elidedText(m_name, Qt::ElideRight, nameW));
    x += nameW + gap;

    // ── Bar track ─────────────────────────────────────────────────────────
    const int barW    = r.width() - pad - nameW - gap - valW - gap - pillW - gap - pad;
    const int barH    = 8;
    const int barY    = (r.height() - barH) / 2;
    QRect trackRect(x, barY, barW, barH);

    // Track background
    QPainterPath trackPath;
    trackPath.addRoundedRect(trackRect, 2, 2);
    p.fillPath(trackPath, QColor(0xeb, 0xe9, 0xe3));
    // Inset shadow on track
    p.setPen(QPen(QColor(0, 0, 0, 15), 1));
    p.drawPath(trackPath);

    // Filled bar
    if (!m_idle && m_score >= 0) {
        const double fill = qBound(0.0, m_score, 1.0);
        QRect barFill(trackRect.x(), trackRect.y(),
                      qRound(trackRect.width() * fill), trackRect.height());
        if (barFill.width() > 0) {
            QPainterPath fillPath;
            fillPath.addRoundedRect(barFill, 2, 2);
            p.fillPath(fillPath, c.bar);
        }
    }

    // Threshold markers at 0.4 and 0.7
    p.setPen(QPen(QColor(0xb9, 0xb3, 0xa6, 128), 1));
    auto marker = [&](double t) {
        int mx = trackRect.x() + qRound(trackRect.width() * t);
        p.drawLine(mx, trackRect.top(), mx, trackRect.bottom());
    };
    marker(0.4);
    marker(0.7);

    x += barW + gap;

    // ── Numeric value ─────────────────────────────────────────────────────
    p.setFont(monoFont);
    p.setPen(c.text);
    QRect valRect(x, 0, valW, r.height());
    const QString valStr = (m_idle || m_score < 0) ? "—"
                         : QString::number(m_score, 'f', 3);
    p.drawText(valRect, Qt::AlignVCenter | Qt::AlignRight, valStr);
    x += valW + gap;

    // ── Status pill ───────────────────────────────────────────────────────
    QFont uiFont("Inter Tight");
    uiFont.setPixelSize(10);
    uiFont.setWeight(QFont::DemiBold);
    if (QFontInfo(uiFont).family().contains("Inter", Qt::CaseInsensitive) == false)
        uiFont.setFamily("Segoe UI");
    p.setFont(uiFont);

    QFontMetrics pillFm(uiFont);
    const QString pillText = c.label;
    const int pillTextW = pillFm.horizontalAdvance(pillText);
    const int pillH     = 18;
    const int pillPadX  = 8;
    const int actualPillW = pillTextW + pillPadX * 2;
    const int pillX = x + (pillW - actualPillW) / 2;
    const int pillY = (r.height() - pillH) / 2;

    QPainterPath pillPath;
    pillPath.addRoundedRect(QRectF(pillX, pillY, actualPillW, pillH), 999, 999);
    p.fillPath(pillPath, c.pillBg);

    p.setPen(c.pillFg);
    p.drawText(QRect(pillX, pillY, actualPillW, pillH),
               Qt::AlignCenter, pillText);
}
