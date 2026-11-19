// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

// PRT-BINK30 (pr-tests, off by default since Build B2 2026-11-19, RYGT_BINK_30HZ=1 turns it on): GT7's pre-rendered
// cutscenes (Bink, decoded on the CPU by the BinkT guest threads) skip frames at 60: the owner's
// logs (2026-11-18) showed 82 to 98 flips per 2 s with 116 to 120 vblanks with no flip ready, the
// video frames held an uneven 1, 2 or 3 vblanks. With the launcher's 30 lock (PERF-075) one flip
// per flip-eligible vblank, clean. While a Bink thread works, flips are locked to 30 per second
// the same way (driver.cpp PresentThread), and the user's rate comes back about 1 s after the
// last Bink activity. The guest's vblank events are left as they are (PERF-075 does not change
// them either).
//
// Activity: the Bink threads look up guest mappings many times a frame while they decode
// (ryty_core_shims.cpp LookupArea; ReportAreaStats lines from BinkT00..02 in the owner's logs).
// Activity is marked once per 64 lookups from a Bink thread, so a stray lookup from an idle one
// does not count. Header only.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "common/thread.h"
#include "common/types.h"

namespace PrtBink30 {

enum Via : u32 {
    ViaLookup = 0,
    ViaCount = 1,
};

inline const char* ViaName(u32 via) {
    return via == ViaLookup ? "mapping lookups" : "?";
}

inline bool On() {
    static const bool on = [] {
        // Build B2: off by default. The championship cutscenes are AvPlayer MP4s, not Bink, and the
        // world map's small Bink video (BinkT04) locked the map to 30.
        const char* env = std::getenv("RYGT_BINK_30HZ");
        return env && env[0] == '1';
    }();
    return on;
}

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct State {
    std::atomic<s64> last_ns{0};      ///< steady clock of the last activity mark, 0 = never
    std::atomic<u64> hits[ViaCount]{}; ///< activity marks by hook
    std::atomic<u32> last_via{0};
    std::mutex name_mutex;
    char name[32]{};                   ///< the last Bink thread that marked activity
};

inline State& GetState() {
    static State state;
    return state;
}

/// Called from the hooks above. Cheap on other threads: their name is checked once, then again
/// every 1024 calls (a thread may be named after it started).
inline void NoteActivity(Via via) {
    if (!On()) {
        return;
    }
    thread_local int state = 0; // 0 unknown, 1 Bink, 2 other
    thread_local u32 calls = 0;
    if (state == 2 && (++calls & 1023) != 0) {
        return;
    }
    if (state != 1) {
        const std::string name = Common::GetCurrentThreadName();
        state = name.rfind("Bink", 0) == 0 ? 1 : 2;
        if (state != 1) {
            return;
        }
        auto& s = GetState();
        std::scoped_lock lk{s.name_mutex};
        std::strncpy(s.name, name.c_str(), sizeof(s.name) - 1);
        s.name[sizeof(s.name) - 1] = '\0';
    }
    if (via == ViaLookup && (++calls & 63) != 0) {
        return;
    }
    auto& s = GetState();
    s.last_ns.store(NowNs(), std::memory_order_relaxed);
    s.hits[via].fetch_add(1, std::memory_order_relaxed);
    s.last_via.store(u32(via), std::memory_order_relaxed);
}

/// Whether a Bink thread marked activity within the last second.
inline bool Active(s64 now_ns) {
    const s64 last = GetState().last_ns.load(std::memory_order_relaxed);
    return last != 0 && now_ns - last < 1'000'000'000;
}

inline std::string ThreadName() {
    auto& s = GetState();
    std::scoped_lock lk{s.name_mutex};
    return std::string(s.name);
}

inline u64 TotalHits() {
    auto& s = GetState();
    u64 total = 0;
    for (u32 i = 0; i < ViaCount; ++i) {
        total += s.hits[i].load(std::memory_order_relaxed);
    }
    return total;
}

} // namespace PrtBink30
