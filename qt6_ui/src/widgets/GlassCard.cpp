#include "GlassCard.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QPainterPath>

GlassCard::GlassCard(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("ContentPanel");
    setAttribute(Qt::WA_TranslucentBackground, false);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void GlassCard::setTitle(const QString& title)   { m_title = title;   update(); }
void GlassCard::setEyebrow(const QString& eyebrow) { m_eyebrow = eyebrow; update(); }

void GlassCard::paintEvent(QPaintEvent* event)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const QRectF r = rect().adjusted(0.5, 0.5, -0.5, -0.5);
    const qreal radius = 6.0;

    // Outer shadow layer 1 (spread, very subtle)
    {
        QPainterPath shadow;
        shadow.addRoundedRect(r.adjusted(0, 1, 0, 0), radius, radius);
        p.fillPath(shadow, QColor(40, 30, 20, 10));
    }

    // Glass fill
    QPainterPath path;
    path.addRoundedRect(r, radius, radius);
    p.fillPath(path, QColor(255, 254, 251, 220));

    // Border
    p.setPen(QPen(QColor(0xd8, 0xd4, 0xcb), 1.0));
    p.drawPath(path);

    // Top highlight (1px inner top edge — warm light)
    p.setPen(QPen(QColor(255, 255, 255, 60), 1.0));
    QPainterPath topEdge;
    topEdge.moveTo(r.left() + radius, r.top() + 1);
    topEdge.lineTo(r.right() - radius, r.top() + 1);
    p.drawPath(topEdge);

    QFrame::paintEvent(event);
}
