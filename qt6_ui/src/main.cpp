#include <QApplication>
#include <QFontDatabase>
#include <QFont>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include "MainWindow.h"
#include "theme/StyleSheet.h"

int main(int argc, char* argv[])
{
    // High-DPI — on by default in Qt6, but be explicit
    QApplication::setHighDpiScaleFactorRoundingPolicy(
        Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QApplication app(argc, argv);
    app.setApplicationName("SUBSTRATE");
    app.setApplicationVersion("0.4.2");
    app.setOrganizationName("SUBSTRATE Scientific");
    app.setOrganizationDomain("substrate.sci");

    // Use Fusion style as the base — it honours QSS faithfully on all platforms.
    app.setStyle(QStyleFactory::create("Fusion"));

    // Set default fonts — Qt will fall back gracefully if not installed.
    {
        QFont uiFont("Inter Tight");
        if (QFontInfo(uiFont).family().contains("Inter", Qt::CaseInsensitive) == false)
            uiFont.setFamily("Segoe UI");
        uiFont.setPixelSize(13);
        app.setFont(uiFont);
    }

    // Apply stylesheet
    app.setStyleSheet(Theme::lightSheet());

    MainWindow w;
    w.show();

    return app.exec();
}
