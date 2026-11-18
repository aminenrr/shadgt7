// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

// PRT-BINK (pr-tests, diagnosis; RYGT_DIAG_BINK=1 turns it on, off by default): GT7's pre-rendered
// cutscenes (Bink 2, decoded on the GPU by compute kernels from the BinkT guest threads) drop
// frames: in a 60 fps capture the picture changed only 13 to 30 times a second. Bink drops a
// frame when its decode is not done by the frame's deadline. This counts, per second while a
// Bink thread submits GPU work: the Bink threads' submits and async compute rings (and how long
// the guest spent inside the call, sceGnmDingDong waits for the GPU's submissions to drain), the
// gap between them, the presented frames, the compute programs dispatched (and those dispatched
// without a pipeline yet), the deferred fences' latency and the draws skipped while a pipeline
// builds. Compute programs first dispatched just after a Bink submit are logged once each: the
// Bink kernels. Header only; nothing runs unless the switch is on.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"

namespace PrtBink {

inline bool On() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_DIAG_BINK");
        return env && env[0] == '1';
    }();
    return on;
}

inline u64 NowUs() {
    return u64(std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

inline void AtomicMax(std::atomic<u64>& value, u64 v) {
    u64 old = value.load(std::memory_order_relaxed);
    while (v > old && !value.compare_exchange_weak(old, v, std::memory_order_relaxed)) {
    }
}

/// Whether the calling thread is one of the guest's Bink threads (its name starts with "Bink").
/// A thread found not to be one is checked again every 1024 calls (named after it started).
inline bool BinkThread() {
    thread_local int state = 0; // 0 unknown, 1 Bink, 2 other
    thread_local u32 calls = 0;
    if (state == 1) {
        return true;
    }
    if (state == 2 && (++calls & 1023) != 0) {
        return false;
    }
    const std::string name = Common::GetCurrentThreadName();
    state = name.rfind("Bink", 0) == 0 ? 1 : 2;
    if (state == 1) {
        LOG_WARNING(Render_Vulkan, "PRT-BINK: guest thread {} submits GPU work", name);
    }
    return state == 1;
}

struct State {
    std::atomic<u64> window_start_us{0};
    std::atomic<u64> flips{0};
    std::atomic<u64> bink_submits{0};
    std::atomic<u64> bink_rings{0};   ///< sceGnmDingDong from a Bink thread (async compute)
    std::atomic<u64> other_submits{0};
    std::atomic<u64> other_rings{0};
    std::atomic<u64> bink_call_us_max{0};
    std::atomic<u64> bink_call_us_total{0};
    std::atomic<u64> bink_gap_us_max{0};
    std::atomic<u64> last_bink_us{0};
    std::atomic<u64> dispatches{0};
    std::atomic<u64> dispatches_no_pipeline{0};
    std::atomic<u64> fences{0};
    std::atomic<u64> fence_us_total{0};
    std::atomic<u64> fence_us_max{0};
    std::atomic<u64> build_skips{0};
    std::mutex mutex;
    std::unordered_map<u64, u32> programs;    ///< this window's compute programs
    std::unordered_set<u64> seen;              ///< every compute program seen
    u32 first_logs = 0;
};

inline State& Get() {
    static State state;
    return state;
}

/// Around a guest's GPU submit (ring: an async compute ring's sceGnmDingDong).
struct SubmitScope {
    bool bink = false;
    u64 start = 0;
    explicit SubmitScope(bool ring) {
        if (!On()) {
            return;
        }
        auto& s = Get();
        bink = BinkThread();
        if (!bink) {
            (ring ? s.other_rings : s.other_submits).fetch_add(1, std::memory_order_relaxed);
            return;
        }
        start = NowUs();
        (ring ? s.bink_rings : s.bink_submits).fetch_add(1, std::memory_order_relaxed);
        const u64 last = s.last_bink_us.exchange(start, std::memory_order_relaxed);
        if (last != 0 && start - last < 2'000'000) {
            AtomicMax(s.bink_gap_us_max, start - last);
        }
    }
    ~SubmitScope() {
        if (bink) {
            auto& s = Get();
            const u64 took = NowUs() - start;
            s.bink_call_us_total.fetch_add(took, std::memory_order_relaxed);
            AtomicMax(s.bink_call_us_max, took);
        }
    }
    SubmitScope(const SubmitScope&) = delete;
    SubmitScope& operator=(const SubmitScope&) = delete;
};

/// A compute dispatch recorded (program 0: none, the pipeline is not there yet).
inline void NoteDispatch(u64 program) {
    if (!On()) {
        return;
    }
    auto& s = Get();
    s.dispatches.fetch_add(1, std::memory_order_relaxed);
    if (program == 0) {
        s.dispatches_no_pipeline.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::scoped_lock lock{s.mutex};
    ++s.programs[program];
    if (s.seen.insert(program).second) {
        const u64 last = s.last_bink_us.load(std::memory_order_relaxed);
        const u64 now = NowUs();
        if (last != 0 && now - last < 100'000 && s.first_logs < 64) {
            ++s.first_logs;
            LOG_WARNING(Render_Vulkan,
                        "PRT-BINK: compute program {:#x} first dispatched {:.1f} ms after a Bink "
                        "thread's GPU submit",
                        program, (now - last) / 1000.0);
        }
    }
}

inline void NoteFence(u64 latency_us) {
    if (!On()) {
        return;
    }
    auto& s = Get();
    s.fences.fetch_add(1, std::memory_order_relaxed);
    s.fence_us_total.fetch_add(latency_us, std::memory_order_relaxed);
    AtomicMax(s.fence_us_max, latency_us);
}

inline void NoteBuildSkip() {
    if (On()) {
        Get().build_skips.fetch_add(1, std::memory_order_relaxed);
    }
}

/// At each presented frame: one line a second while Bink threads submit GPU work.
inline void Tick() {
    if (!On()) {
        return;
    }
    auto& s = Get();
    s.flips.fetch_add(1, std::memory_order_relaxed);
    const u64 now = NowUs();
    u64 start = s.window_start_us.load(std::memory_order_relaxed);
    if (start == 0) {
        s.window_start_us.store(now, std::memory_order_relaxed);
        return;
    }
    if (now - start < 1'000'000 ||
        !s.window_start_us.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 flips = s.flips.exchange(0);
    const u64 bink_submits = s.bink_submits.exchange(0);
    const u64 bink_rings = s.bink_rings.exchange(0);
    const u64 other_submits = s.other_submits.exchange(0);
    const u64 other_rings = s.other_rings.exchange(0);
    const u64 call_max = s.bink_call_us_max.exchange(0);
    const u64 call_total = s.bink_call_us_total.exchange(0);
    const u64 gap_max = s.bink_gap_us_max.exchange(0);
    const u64 dispatches = s.dispatches.exchange(0);
    const u64 no_pipeline = s.dispatches_no_pipeline.exchange(0);
    const u64 fences = s.fences.exchange(0);
    const u64 fence_total = s.fence_us_total.exchange(0);
    const u64 fence_max = s.fence_us_max.exchange(0);
    const u64 skips = s.build_skips.exchange(0);
    std::vector<std::pair<u64, u32>> top;
    {
        std::scoped_lock lock{s.mutex};
        top.assign(s.programs.begin(), s.programs.end());
        s.programs.clear();
    }
    if (bink_submits + bink_rings == 0) {
        return; // no video playing
    }
    std::sort(top.begin(), top.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string programs;
    for (size_t i = 0; i < top.size() && i < 6; ++i) {
        programs += fmt::format("{}{:#x} x{}", i ? ", " : "", top[i].first, top[i].second);
    }
    const double seconds = (now - start) / 1e6;
    LOG_WARNING(Render_Vulkan,
                "PRT-BINK {:.1f} s: {} frames presented; Bink GPU submits {} and async compute "
                "rings {} (~video frames decoded), gap max {:.1f} ms, in the submit call max "
                "{:.2f} ms, total {:.1f} ms; other submits {}, rings {}; compute dispatches {} "
                "({} without a pipeline yet), top: {}; deferred fences {} avg {:.2f} ms max "
                "{:.2f} ms; draws skipped while a pipeline builds {}",
                seconds, flips, bink_submits, bink_rings, gap_max / 1000.0, call_max / 1000.0,
                call_total / 1000.0, other_submits, other_rings, dispatches, no_pipeline,
                programs.empty() ? std::string{"none"} : programs, fences,
                fences ? fence_total / 1000.0 / fences : 0.0, fence_max / 1000.0, skips);
}

} // namespace PrtBink
