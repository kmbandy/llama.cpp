#include "mt-semantic.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>

namespace mt {

namespace {

inline float dot(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

void SemanticIndex::add_fingerprint(std::vector<llama_pos> positions,
                                     std::vector<float>     embedding,
                                     Tier                   tier) {
    std::lock_guard<std::mutex> lk(mu_);

    if (fingerprints_.size() >= kMaxFingerprints) {
        // Evict the oldest entry (lowest turn). The vector is mostly
        // append-ordered; in practice the front is the oldest.
        size_t oldest = 0;
        uint64_t oldest_turn = fingerprints_.front().turn;
        for (size_t i = 1; i < fingerprints_.size(); ++i) {
            if (fingerprints_[i].turn < oldest_turn) {
                oldest_turn = fingerprints_[i].turn;
                oldest      = i;
            }
        }
        fingerprints_.erase(fingerprints_.begin() + oldest);
    }

    Fingerprint fp;
    fp.positions = std::move(positions);
    fp.embedding = std::move(embedding);
    fp.tier      = tier;
    fp.turn      = ++next_turn_;
    fingerprints_.push_back(std::move(fp));
}

std::vector<SemanticIndex::Hint> SemanticIndex::score(
        const std::vector<float> & query,
        int                        top_k,
        float                      threshold) const {
    std::vector<Hint> out;
    if (query.empty() || top_k <= 0) return out;

    std::lock_guard<std::mutex> lk(mu_);
    if (fingerprints_.empty()) return out;

    // Compute scores once, sort, threshold, top-k.
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(fingerprints_.size());
    for (size_t i = 0; i < fingerprints_.size(); ++i) {
        const float s = dot(query, fingerprints_[i].embedding);
        scored.emplace_back(s, i);
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });

    out.reserve((size_t) top_k);
    for (const auto & [s, idx] : scored) {
        if (s < threshold) break;
        Hint h;
        h.positions = fingerprints_[idx].positions;
        h.score     = s;
        h.tier      = fingerprints_[idx].tier;
        out.push_back(std::move(h));
        if ((int) out.size() >= top_k) break;
    }
    return out;
}

void SemanticIndex::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    fingerprints_.clear();
    next_turn_ = 0;
}

size_t SemanticIndex::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return fingerprints_.size();
}

// ---------------------------------------------------------------------------
// Persistence — simple binary format.
//
//   uint32  magic     = 0x4D544649   ("MTFI")
//   uint32  version   = 1
//   uint64  n_fps
//   for each fp:
//     uint8   tier
//     uint64  turn
//     uint32  n_positions
//     llama_pos[n_positions]    (sizeof(llama_pos) is platform-stable;
//                                we store as int32_t LE for portability)
//     uint32  embedding_dim
//     float[embedding_dim]
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t kFingerprintFileMagic   = 0x4D544649;
constexpr uint32_t kFingerprintFileVersion = 1;
}

bool SemanticIndex::save_to_disk(const std::string & path) const {
    std::lock_guard<std::mutex> lk(mu_);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        LLAMA_LOG_WARN("mt::SemanticIndex::save: open(%s) failed\n", path.c_str());
        return false;
    }

    auto write_u32 = [&](uint32_t v) { f.write((const char *) &v, sizeof(v)); };
    auto write_u64 = [&](uint64_t v) { f.write((const char *) &v, sizeof(v)); };
    auto write_u8  = [&](uint8_t  v) { f.write((const char *) &v, sizeof(v)); };

    write_u32(kFingerprintFileMagic);
    write_u32(kFingerprintFileVersion);
    write_u64((uint64_t) fingerprints_.size());

    for (const auto & fp : fingerprints_) {
        write_u8((uint8_t) fp.tier);
        write_u64(fp.turn);
        write_u32((uint32_t) fp.positions.size());
        for (llama_pos p : fp.positions) {
            int32_t v = (int32_t) p;
            f.write((const char *) &v, sizeof(v));
        }
        write_u32((uint32_t) fp.embedding.size());
        f.write((const char *) fp.embedding.data(),
                (std::streamsize)(fp.embedding.size() * sizeof(float)));
    }

    return f.good();
}

bool SemanticIndex::load_from_disk(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        // Not an error — file just doesn't exist yet on first run.
        return false;
    }

    auto read_u32 = [&](uint32_t & v) -> bool {
        f.read((char *) &v, sizeof(v));
        return (bool) f;
    };
    auto read_u64 = [&](uint64_t & v) -> bool {
        f.read((char *) &v, sizeof(v));
        return (bool) f;
    };
    auto read_u8 = [&](uint8_t & v) -> bool {
        f.read((char *) &v, sizeof(v));
        return (bool) f;
    };

    uint32_t magic = 0, version = 0;
    if (!read_u32(magic) || !read_u32(version)) return false;
    if (magic != kFingerprintFileMagic || version != kFingerprintFileVersion) {
        LLAMA_LOG_WARN("mt::SemanticIndex::load: bad header (magic=%08x ver=%u) in %s\n",
                       magic, version, path.c_str());
        return false;
    }

    uint64_t n_fps = 0;
    if (!read_u64(n_fps)) return false;
    if (n_fps > kMaxFingerprints) {
        LLAMA_LOG_WARN("mt::SemanticIndex::load: %lu fingerprints > cap %zu — truncating\n",
                       (unsigned long) n_fps, kMaxFingerprints);
        n_fps = kMaxFingerprints;
    }

    std::vector<Fingerprint> loaded;
    loaded.reserve((size_t) n_fps);
    for (uint64_t i = 0; i < n_fps; ++i) {
        uint8_t  tier = 0;
        uint64_t turn = 0;
        uint32_t n_pos = 0, emb_dim = 0;
        if (!read_u8(tier) || !read_u64(turn) || !read_u32(n_pos)) return false;

        Fingerprint fp;
        fp.tier = (Tier) tier;
        fp.turn = turn;
        fp.positions.resize(n_pos);
        for (uint32_t j = 0; j < n_pos; ++j) {
            int32_t v = 0;
            f.read((char *) &v, sizeof(v));
            if (!f) return false;
            fp.positions[j] = (llama_pos) v;
        }
        if (!read_u32(emb_dim)) return false;
        fp.embedding.resize(emb_dim);
        f.read((char *) fp.embedding.data(),
               (std::streamsize)(emb_dim * sizeof(float)));
        if (!f) return false;
        loaded.push_back(std::move(fp));
    }

    std::lock_guard<std::mutex> lk(mu_);
    fingerprints_ = std::move(loaded);
    next_turn_    = fingerprints_.empty() ? 0 : fingerprints_.back().turn;
    return true;
}

}  // namespace mt
