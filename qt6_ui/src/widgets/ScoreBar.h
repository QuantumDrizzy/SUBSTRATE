#pragma once
#include <QWidget>

// ScoreBar — one row in the layer score list.
// Renders: name | bar track (with 0.4/0.7 threshold markers) | value | pill.
// Matches components-score-bar.html exactly.
class ScoreBar : public QWidget
{
    Q_OBJECT
public:
    explicit ScoreBar(QWidget* parent = nullptr);

    void setLayer(const QString& name, double score, bool idle = false);

    QSize sizeHint() const override { return QSize(400, 32); }
    QSize minimumSizeHint() const override { return QSize(200, 32); }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QString m_name;
    double  m_score  = -1.0;
    bool    m_idle   = false;

    struct Colors {
        QColor bar, text, pillBg, pillFg;
        QString label;
    };
    Colors colorsForScore() const;
};
