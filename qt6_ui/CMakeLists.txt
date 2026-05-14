cmake_minimum_required(VERSION 3.22)
project(SUBSTRATE VERSION 0.4.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

qt_standard_project_setup()

set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/panels/DashboardPanel.cpp
    src/panels/LayerDetailPanel.cpp
    src/widgets/ScoreBar.cpp
    src/widgets/GlassCard.cpp
)

set(HEADERS
    src/MainWindow.h
    src/theme/StyleSheet.h
    src/panels/DashboardPanel.h
    src/panels/LayerDetailPanel.h
    src/widgets/ScoreBar.h
    src/widgets/GlassCard.h
)

qt_add_executable(substrate ${SOURCES} ${HEADERS})

target_include_directories(substrate PRIVATE src)

target_link_libraries(substrate PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

# Windows: high-DPI, no console window
if(WIN32)
    set_target_properties(substrate PROPERTIES WIN32_EXECUTABLE TRUE)
    target_compile_definitions(substrate PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Deployment helper
if(WIN32)
    add_custom_command(TARGET substrate POST_BUILD
        COMMAND "${Qt6_DIR}/../../../bin/windeployqt.exe"
                "$<TARGET_FILE:substrate>"
        COMMENT "Running windeployqt"
        VERBATIM
    )
endif()
