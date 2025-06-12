#include "../timer/PerformanceTimer.h"
#include <thread>
#include <iostream>

int main() {
    auto& tm = PerformanceTimer::instance();
    tm.start("任务A");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    tm.end("任务A");

    for (int i = 0; i < 3; ++i) {
        tm.start("循环任务");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        tm.end("循环任务");
    }

    tm.printAll();
    return 0;
}

