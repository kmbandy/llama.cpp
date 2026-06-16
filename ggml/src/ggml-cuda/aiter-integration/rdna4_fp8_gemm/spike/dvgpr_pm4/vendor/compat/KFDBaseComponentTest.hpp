// compat/KFDBaseComponentTest.hpp
//
// Minimal SHIM that satisfies what the vendored kfdtest PM4Packet.cpp actually
// uses from kfdtest's heavy KFDBaseComponentTest.hpp, WITHOUT pulling in gtest,
// libdrm, ShaderStore, the whole test framework, etc.
//
// PM4Packet.cpp only needs two things from that header:
//   * the KfdFamilyId enum values FAMILY_AI / FAMILY_NV (family branching), and
//   * bool hsakmt_is_dgpu()   (ATC / dGPU-vs-APU packet flags).
//
// The KfdFamilyId enum is copied verbatim (same ordering => same numeric
// values) from kfdtest src/KFDTestFlags.hpp @ ROCR-Runtime ba56a24c, so
// FAMILY_GFX12 == 12, FAMILY_NV == 10, FAMILY_AI == 5 exactly as upstream.
#ifndef __KFD_COMPAT_BASECOMPONENT_SHIM__
#define __KFD_COMPAT_BASECOMPONENT_SHIM__

#include "hsakmt/hsakmt.h"

// Verbatim from kfdtest KFDTestFlags.hpp (enum KfdFamilyId), ba56a24c.
enum KfdFamilyId {
    FAMILY_UNKNOWN = 0,
    FAMILY_CI,    // Sea Islands
    FAMILY_KV,    // Kaveri / Kabini
    FAMILY_VI,    // Volcanic Islands
    FAMILY_CZ,    // Carrizo
    FAMILY_AI,    // Arctic Islands      (== 5)
    FAMILY_RV,    // Raven
    FAMILY_AR,    // Arcturus
    FAMILY_AL,    // Aldebaran
    FAMILY_AV,    // Aqua Vanjaram
    FAMILY_NV,    // Navi10              (== 10)
    FAMILY_GFX11, // GFX11
    FAMILY_GFX12, // GFX12               (== 12)  <- gfx1201 / R9700
};

// Defined in pm4_dispatch.cpp. The vendored encoder calls this to pick ATC /
// memory-controller packet flags; on the headless dGPU R9700 it returns true.
bool hsakmt_is_dgpu();

#endif  // __KFD_COMPAT_BASECOMPONENT_SHIM__
