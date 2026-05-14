#pragma once
#include <QWidget>
#include <QString>
#include <array>

struct LayerData {
    QString id;
    QString name;
    QString type;
    double  score;    // -1 = idle/no data
    double  mu;
    double  sigma;
    bool    running;
};

// DashboardPanel — 2×3 grid of physics-layer glass cards.
// Each card: eyebrow (layer id) | big mono score | score bar | status pill | metadata.
class LayerCard;

class DashboardPanel : public QWidget
{
    Q_OBJECT
public:
    explicit DashboardPanel(QWidget* parent = nullptr);

    void updateLayer(int index, const LayerData& data);

signals:
    void layerClicked(int index);

private:
    std::array<LayerCard*, 6> m_cards{};
};
