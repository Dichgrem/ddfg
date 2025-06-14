#include "PerformanceMonitor.h"
#include <iostream>
#include <iomanip>
#include <algorithm> // For std::sort

PerformanceMonitor& PerformanceMonitor::getInstance() {
    static PerformanceMonitor instance;
    return instance;
}

void PerformanceMonitor::startTask(const std::string& task_name) {
    TaskStats& stats = task_data_[task_name];
    if (stats.is_running) {
        // std::cerr << "[WARN] Task '" << task_name << "' already started. Ignoring consecutive start." << std::endl;
        return; // 防止重复开始导致错误
    }
    stats.current_start_time = Clock::now();
    stats.is_running = true;
}

void PerformanceMonitor::stopTask(const std::string& task_name) {
    TaskStats& stats = task_data_[task_name];
    if (!stats.is_running) {
        // std::cerr << "[WARN] Task '" << task_name << "' not started. Ignoring stop." << std::endl;
        return; // 防止停止未开始的任务
    }
    TimePoint end_time = Clock::now();
    DurationNs duration = std::chrono::duration_cast<DurationNs>(end_time - stats.current_start_time);
    stats.addDuration(duration.count());
    stats.is_running = false;
}

void PerformanceMonitor::startFrame() {
    frame_start_time_ = Clock::now();
}

void PerformanceMonitor::stopFrame() {
    TimePoint end_time = Clock::now();
    DurationNs duration = std::chrono::duration_cast<DurationNs>(end_time - frame_start_time_);
    frame_durations_ns_.push_back(duration.count());
}

void PerformanceMonitor::printReport() const {
    if (frame_durations_ns_.empty() && task_data_.empty()) {
        std::cout << "No performance data to report." << std::endl;
        return;
    }

    std::cout << "\n--- Performance Report ---\n";
    std::cout << std::fixed << std::setprecision(2);

    if (!frame_durations_ns_.empty()) {
        long long total_frame_ns = 0;
        long long min_frame_ns = -1;
        long long max_frame_ns = -1;
        for (long long ns : frame_durations_ns_) {
            total_frame_ns += ns;
            if (min_frame_ns == -1 || ns < min_frame_ns) {
                min_frame_ns = ns;
            }
            if (ns > max_frame_ns) {
                max_frame_ns = ns;
            }
        }

        double avg_frame_ms = static_cast<double>(total_frame_ns) / frame_durations_ns_.size() / 1000000.0;
        double min_frame_ms = static_cast<double>(min_frame_ns) / 1000000.0;
        double max_frame_ms = static_cast<double>(max_frame_ns) / 1000000.0;
        double fps = (avg_frame_ms > 0) ? (1000.0 / avg_frame_ms) : 0.0;

        std::cout << "Total Frames Processed: " << frame_durations_ns_.size() << " frames\n";
        std::cout << "Overall Frame Processing:\n";
        std::cout << "  Min: " << min_frame_ms << " ms\n";
        std::cout << "  Max: " << max_frame_ms << " ms\n";
        std::cout << "  Avg: " << avg_frame_ms << " ms\n";
        std::cout << "  FPS: " << fps << " FPS\n\n";
    }

    std::vector<std::pair<std::string, const TaskStats*>> sorted_tasks;
    for (const auto& pair : task_data_) {
        sorted_tasks.push_back({pair.first, &pair.second});
    }

    // 按平均耗时降序排序，找出最耗时的任务
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), [](const auto& a, const auto& b) {
        return a.second->getAverageMs() > b.second->getAverageMs();
    });

    std::cout << "Task Breakdown:\n";
    for (const auto& pair : sorted_tasks) {
        const std::string& task_name = pair.first;
        const TaskStats& stats = *pair.second;
        if (stats.num_runs > 0) {
            std::cout << "  " << std::setw(20) << std::left << task_name << ": "
                      << "Runs: " << std::setw(6) << stats.num_runs
                      << " Avg: " << std::setw(8) << stats.getAverageMs() << "ms"
                      << " Min: " << std::setw(8) << (double)stats.min_ns / 1000000.0 << "ms"
                      << " Max: " << std::setw(8) << (double)stats.max_ns / 1000000.0 << "ms\n";
        }
    }
    std::cout << "---------------------------\n";
}

void PerformanceMonitor::reset() {
    task_data_.clear();
    frame_durations_ns_.clear();
    std::cout << "Performance data reset.\n";
}
