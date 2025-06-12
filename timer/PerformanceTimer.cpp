#include "PerformanceTimer.h"
#include <iostream>

PerformanceTimer& PerformanceTimer::instance() {
    static PerformanceTimer inst;
    return inst;
}

void PerformanceTimer::start(const std::string& name) {
    _starts[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceTimer::end(const std::string& name) {
    auto end = std::chrono::high_resolution_clock::now();
    auto startIt = _starts.find(name);
    if (startIt == _starts.end()) return;

    double diff = std::chrono::duration<double, std::milli>(end - startIt->second).count();
    auto& s = _data[name];
    if (s.count++ == 0) {
        s.min = s.max = diff;
    } else {
        s.min = std::min(s.min, diff);
        s.max = std::max(s.max, diff);
    }
    s.total += diff;
}

void PerformanceTimer::printAll() const {
    for (auto& [name, s] : _data) {
        std::cout << "Timer Name: " << name << "\n"
                  << "Count: " << s.count << "\n"
                  << "Min: " << s.min << " ms\n"
                  << "Max: " << s.max << " ms\n"
                  << "Avg: " << (s.total / s.count) << " ms\n"
                  << "--------------------\n";
    }
}

