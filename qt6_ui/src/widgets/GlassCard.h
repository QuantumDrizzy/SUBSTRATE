#pragma once
#include <QFrame>
#include <QPainter>
#include <QPainterPath>

// GlassCard — frosted-glass container panel.
// Qt doesn't support CSS backdrop-filter, so we paint a warm semi-transparent
// fill + 1px border + two-layer box shadow to approximate the glass aesthetic.
class GlassCard : public QFrame
{
    Q_OBJECT
public:
    explicit GlassCard(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setEyebrow(const QString& eyebrow);

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QString m_title;
    QString m_eyebrow;
};
