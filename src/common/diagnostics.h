// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Diagnostics in the build (CMake's SHADGT_DIAGNOSTICS: on in Debug and RelWithDebInfo, off in
// Release unless asked for). Without them, a diagnostic environment switch reads as unset, so the
// code it guards is dropped at compile time and Release behaves like a run with none of them set.
// Switches that change what the game shows or how it runs keep std::getenv in every build.

#include <cstdlib>

namespace Common {

#ifdef SHADGT_DIAGNOSTICS
inline constexpr bool DiagnosticsBuilt = true;

/// std::getenv for a diagnostic switch (RYGT_DIAG_*, RYGT_AB, the F1 dumps, ...).
inline const char* DiagEnv(const char* name) {
    return std::getenv(name);
}
#else
inline constexpr bool DiagnosticsBuilt = false;

constexpr const char* DiagEnv(const char*) {
    return nullptr;
}
#endif

/// PRT-LOGS (Build B4, pr-tests): the old routine log lines (each pipeline built, per-texture
/// overlap notes, the every-2-s perf lines of past investigations) only with RYGT_LOG_VERBOSE=1.
/// Read once; the launcher's "Verbose logs" toggle sets it. Off by default.
inline bool VerboseLogs() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_LOG_VERBOSE");
        return env && env[0] == '1';
    }();
    return on;
}

/// PRT-PREWARM-ONLY (Build B5, pr-tests): RYGT_PREWARM_ONLY=1, set by the launcher's "Precompile
/// graphics shaders" step. The renderer starts as in game (same Vulkan instance and device, so
/// the driver's cache keys match), builds the recorded pipelines and compute recipes on every
/// core, saves the pipeline cache and ends the process before the game's code runs.
inline bool PrewarmOnly() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_PREWARM_ONLY");
        return env && env[0] == '1';
    }();
    return on;
}

} // namespace Common
