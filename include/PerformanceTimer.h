#ifndef PERFORMANCE_TIMER_H
#define PERFORMANCE_TIMER_H

#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <vector>
#include <limits>
#include <iomanip>

class TimerManager {
public:
    // 获取单例实例
    static TimerManager& getInstance();

    // 开始计时
    void start(const std::string& name);

    // 结束计时
    void end(const std::string& name);

    // 打印所有计时器摘要
    void printAllSummaries() const;

    // 不允许拷贝和赋值
    TimerManager(const TimerManager&) = delete;
    TimerManager& operator=(const TimerManager&) = delete;

private:
    TimerManager() = default; // 私有构造函数

    struct TimerStats {
        std::chrono::high_resolution_clock::time_point start_time;
        long long count = 0;
        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = std::numeric_limits<double>::min();
    };

    std::map<std::string, TimerStats> timers_;
    std::vector<std::string> timer_order_; // 保持插入顺序
};

#endif // PERFORMANCE_TIMER_H
