// Segment layout shared with peer/segment.py. UNVALIDATED: the device kernels
// do not map this segment. The Python sidecar polls it.
#pragma once
#include <cstdint>

namespace inkling_proto {
constexpr int kHidden = 4608;
constexpr int kTopk = 8;
constexpr int kMaxTokens = 64;
constexpr uint64_t kMagic = 0x52454550324B4E49ull;  // bytes of "INK2PEER", little endian
constexpr uint64_t kVersion = 1;
constexpr uint64_t kOffMagic = 0;
constexpr uint64_t kOffVersion = 8;
constexpr uint64_t kOffReq = 16;
constexpr uint64_t kOffResp = 24;
constexpr uint64_t kOffLayer = 32;
constexpr uint64_t kOffQuant = 36;
constexpr uint64_t kOffT = 40;
constexpr uint64_t kOffErr = 44;
constexpr uint64_t kOffX = 4096;
constexpr uint64_t kOffIds = 593920;
constexpr uint64_t kOffW = 598016;
constexpr uint64_t kOffY = 602112;
constexpr uint64_t kSegBytes = 1191936;
}  // namespace inkling_proto
