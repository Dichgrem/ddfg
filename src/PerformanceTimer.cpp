#include "PerformanceTimer.h"

TimerManager& TimerManager::getInstance() {
    static TimerManager instance;
    return instance;
}

void TimerManager::start(const std::string& name) {
    if (timers_.find(name) == timers_.end()) {
        timers_[name] = TimerStats{};
        timer_order_.push_back(name);
    }
    timers_[name].start_time = std::chrono::high_resolution_clock::now();
}

void TimerManager::end(const std::string& name) {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        std::cerr << "Warning: Timer '" << name << "' was ended but never started." << std::endl;
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - it->second.start_time;
    double duration = elapsed_ms.count();

    it->second.count++;
    it->second.total_ms += duration;
    if (duration < it->second.min_ms) it->second.min_ms = duration;
    if (duration > it->second.max_ms) it->second.max_ms = duration;
}

void TimerManager::printAllSummaries() const {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "             Performance Summary" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    for (const auto& name : timer_order_) {
        const auto& stats = timers_.at(name);
        if (stats.count == 0) continue;

        double avg_ms = stats.total_ms / stats.count;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Timer Name: " << name << std::endl;
        std::cout << "  Count: " << stats.count << std::endl;
        std::cout << "  Min:   " << stats.min_ms << " ms" << std::endl;
        std::cout << "  Max:   " << stats.max_ms << " ms" << std::endl;
        std::cout << "  Avg:   " << avg_ms << " ms" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
    }
}
