#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric> // For std::accumulate
#include <iostream> // For std::cerr in ScopedPerformanceMonitor (optional, but good for warnings)

class PerformanceMonitor {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using DurationNs = std::chrono::nanoseconds;

    // 获取单例实例
    static PerformanceMonitor& getInstance();

    // 开始一个任务计时
    void startTask(const std::string& task_name);

    // 停止一个任务计时
    void stopTask(const std::string& task_name);

    // 记录一帧的开始时间
    void startFrame();

    // 记录一帧的结束时间
    void stopFrame();

    // 打印所有统计信息
    void printReport() const;

    // 清除所有统计数据
    void reset();

private:
    PerformanceMonitor() = default; // 私有构造函数，实现单例
    ~PerformanceMonitor() = default; // 私有析构函数
    PerformanceMonitor(const PerformanceMonitor&) = delete; // 禁用拷贝构造
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete; // 禁用赋值操作

    struct TaskStats {
        std::vector<long long> durations_ns; // 存储每次运行的纳秒时长
        TimePoint current_start_time;       // 当前任务的开始时间
        bool is_running = false;            // 任务是否正在计时
        long long min_ns = -1;              // 最小耗时
        long long max_ns = -1;              // 最大耗时
        long long total_ns = 0;             // 总耗时
        long long num_runs = 0;             // 运行次数

        void addDuration(long long ns) {
            durations_ns.push_back(ns);
            total_ns += ns;
            num_runs++;
            if (min_ns == -1 || ns < min_ns) {
                min_ns = ns;
            }
            if (ns > max_ns) {
                max_ns = ns;
            }
        }

        double getAverageMs() const {
            if (num_runs == 0) return 0.0;
            return static_cast<double>(total_ns) / num_runs / 1000000.0; // 纳秒转毫秒
        }
    };

    std::unordered_map<std::string, TaskStats> task_data_;
    TimePoint frame_start_time_;
    std::vector<long long> frame_durations_ns_; // 存储每帧的纳秒时长
};


// -----------------------------------------------------------
// 以下是 ScopedPerformanceMonitor 和相关的宏，它们现在在
// PerformanceMonitor 类定义之后，确保 PerformanceMonitor 已被声明。
// -----------------------------------------------------------

// 方便的宏，用于在代码中标记性能统计点
#define PM_START(task_name) PerformanceMonitor::getInstance().startTask(task_name);
#define PM_STOP(task_name) PerformanceMonitor::getInstance().stopTask(task_name);

// 自动停止的辅助类，用于 RAII 风格的时间测量
class ScopedPerformanceMonitor {
public:
    ScopedPerformanceMonitor(const std::string& task_name) : task_name_(task_name) {
        PerformanceMonitor::getInstance().startTask(task_name_);
    }

    ~ScopedPerformanceMonitor() {
        PerformanceMonitor::getInstance().stopTask(task_name_);
    }

private:
    std::string task_name_;
};

// 方便的宏，用于自动管理任务的开始和结束
#define PM_SCOPED(task_name) ScopedPerformanceMonitor _scoped_pm_##task_name(#task_name);
