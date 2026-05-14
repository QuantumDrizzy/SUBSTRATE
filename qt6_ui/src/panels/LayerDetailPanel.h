#pragma once
#include <QWidget>
#include <QLabel>
#include "DashboardPanel.h"  // LayerData, QString

class QLineEdit;
class QComboBox;
class ScoreBar;

// LayerDetailPanel — right-side inspector + solver config.
// Matches the Inspector + config form from components.jsx exactly.
class LayerDetailPanel : public QWidget
{
    Q_OBJECT
public:
    explicit LayerDetailPanel(QWidget* parent = nullptr);

    void showLayer(const LayerData& data);
    void clearLayer();

private:
    void buildDetailSection();
    void buildConfigSection();

    // Detail section
    QWidget*   m_detailSection = nullptr;
    QLabel*    m_idLabel       = nullptr;
    QLabel*    m_readout       = nullptr;
    QLabel*    m_pill          = nullptr;
    QLabel*    m_kvType        = nullptr;
    QLabel*    m_kvMu          = nullptr;
    QLabel*    m_kvSigma       = nullptr;
    QLabel*    m_kvVisible     = nullptr;
    QLabel*    m_kvLocked      = nullptr;
    ScoreBar*  m_bar           = nullptr;
    QLabel*    m_emptyMsg      = nullptr;

    // Config section
    QComboBox* m_algCombo = nullptr;
    QLineEdit* m_tolEdit  = nullptr;
    QLineEdit* m_iterEdit = nullptr;
    QLineEdit* m_dampEdit = nullptr;
};
