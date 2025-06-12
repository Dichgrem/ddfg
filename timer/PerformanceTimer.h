#ifndef PERFORMANCE_TIMER_H
#define PERFORMANCE_TIMER_H

#include <string>
#include <chrono>
#include <map>

struct Summary {
    size_t count = 0;
    double total = 0;
    double min = 0;
    double max = 0;
};

class PerformanceTimer {
public:
    static PerformanceTimer& instance();
    void start(const std::string& name);
    void end(const std::string& name);
    void printAll() const;

private:
    PerformanceTimer() = default;
    std::map<std::string, Summary> _data;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> _starts;
};

#endif // PERFORMANCE_TIMER_H

