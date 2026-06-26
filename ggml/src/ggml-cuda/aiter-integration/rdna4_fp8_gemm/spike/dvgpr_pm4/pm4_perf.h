// pm4_perf.h  (MAD-305 in-ring GL2C perfcounter rig)
//
// Two offset-INDEPENDENT PM4 packet encoders needed to read GL2C (RDNA L2)
// hardware perfcounters from WITHIN our own libhsakmt KFD compute ring -- the
// ONLY way to instrument our raw-PM4 kernels, which are invisible to every
// ROCr-based profiler (rocprofv3 / RGP / rocprofiler-sdk device counting all
// read 0 for a KFD-queue dispatch).
//
//   PM4WriteRegPacket : WRITE_DATA -> MEM_MAPPED_REGISTER. Writes one MMIO
//                       register by absolute DWORD offset. Used to program
//                       GRBM_GFX_INDEX (instance bank-select), the
//                       GL2C_PERFCOUNTER*_SELECT event registers, and
//                       CP_PERFMON_CNTL (reset/start/stop).
//   PM4CopyDataPacket : COPY_DATA register->memory. Reads a perfcounter
//                       LO/HI (or a 64-bit pair) into a CPU-visible result BO.
//
// Both subclass the vendored kfdtest PM4Packet so they drop straight into the
// existing RingPlace(ring, pkt) path. NO register offsets are baked in here --
// the caller passes absolute DWORD offsets (those come from the verified
// gfx12/gc_12_0_0 table). src_sel / engine_sel are caller-supplied so the exact
// read path (perfcounter vs mem-mapped-register) and engine (ME/PFP) are
// call-site decisions, not rewrites.
//
// Encodings transcribed from the PM4 type-3 spec (gfx9..gfx12 stable):
//   WRITE_DATA  (IT=0x37): dst_sel@[11:8], addr_incr@[16], wr_confirm@[20],
//                          engine_sel@[31:30]; then dst_lo, dst_hi, data[].
//                          For dst_sel=MEM_MAPPED_REGISTER the "address" field
//                          is the register DWORD offset (NOT a byte addr, NOT
//                          >>2). We reuse the vendored PM4WRITE_DATA_CI struct.
//   COPY_DATA   (IT=0x40): src_sel@[3:0], dst_sel@[11:8], count_sel@[16],
//                          wr_confirm@[20], engine_sel@[31:30]; then
//                          src_lo/reg_off, src_hi, dst_lo, dst_hi. Hand-encoded
//                          (no vendored struct).
#ifndef RDNA4_DVGPR_PM4_PERF_H
#define RDNA4_DVGPR_PM4_PERF_H

#include <cstddef>   // offsetof, NULL (used by the vendored PM4Packet.hpp inlines)
#include <cstdint>
#include <cstring>

#include "PM4Packet.hpp"   // BasePacket / PM4Packet, vendored encoders + enums

// COPY_DATA src_sel / dst_sel select values (PM4 type-3, gfx9..gfx12 stable).
enum CopyDataSrcSel {
    COPY_DATA_SRC_MEM_MAPPED_REG = 0,  // read an MMIO register by dword offset
    COPY_DATA_SRC_MEMORY         = 1,  // 32-bit memory / GPU VA
    COPY_DATA_SRC_TC_L2          = 2,
    COPY_DATA_SRC_GDS            = 3,
    COPY_DATA_SRC_PERFCOUNTERS   = 4,  // read via the perfcounter access path
    COPY_DATA_SRC_IMMEDIATE      = 5,
    COPY_DATA_SRC_ATOMIC_RETURN  = 6,
};
enum CopyDataDstSel {
    COPY_DATA_DST_MEM_MAPPED_REG = 0,
    COPY_DATA_DST_MEMORY         = 1,  // write to memory / GPU VA (sync, via ME)
    COPY_DATA_DST_TC_L2          = 2,
    COPY_DATA_DST_GDS            = 3,
    COPY_DATA_DST_PERFCOUNTERS   = 4,
    COPY_DATA_DST_MEMORY_ASYNC   = 5,
};
enum CopyDataCountSel { COPY_DATA_COUNT_32BIT = 0, COPY_DATA_COUNT_64BIT = 1 };
enum CopyDataEngineSel { COPY_DATA_ENGINE_ME = 0, COPY_DATA_ENGINE_PFP = 1 };

// ===========================================================================
// gfx12 / gfx1201 (RDNA4, R9700) GL2C perfcounter register map -- VERIFIED.
//
// Every value below was reproduced from a PRIMARY source, by hand, and the
// independent sources agree exactly (a wrong offset can hang the compute queue,
// and this GPU drives the displays, so nothing here is guessed):
//   * register absolute DWORD offsets : Mesa src/amd/registers/gfx12.json,
//     parsed locally (and cross-checked vs kernel gc_12_0_0_offset.h reg+BASE_IDX
//     1, GC segment base 0xA000).
//   * field bit-masks : kernel gc_12_0_0_sh_mask.h (grepped locally).
//   * GL2C PERF_SEL event ids : ROCm rocprofiler-sdk counter_defs.yaml
//     (architectures: gfx1201, block: GL2C), parsed locally.
// All these registers live in UCONFIG space (dword >= 0xc000). WRITE_DATA ->
// MEM_MAPPED_REGISTER takes the ABSOLUTE dword offset (these values directly).
// ===========================================================================

// --- register absolute DWORD offsets ---
static const uint32_t mmGRBM_GFX_INDEX           = 0xc200;
static const uint32_t mmCP_PERFMON_CNTL          = 0xd808;  // (= regCP_PERFMON_CNTL_1)
static const uint32_t mmGL2C_PERFCOUNTER0_SELECT = 0xdb80;  // SELECT0..3 = +0,+2,+4,+6
static const uint32_t mmGL2C_PERFCOUNTER0_LO     = 0xd380;  // counter i LO/HI = 0xd380 + 2*i (+1 = HI)
static const uint32_t GL2C_SELECT_STRIDE = 2;
static const uint32_t GL2C_RESULT_STRIDE = 2;

// --- GRBM_GFX_INDEX fields (gc_12_0_0_sh_mask.h) ---
//   INSTANCE_INDEX[6:0] (0x7f) · SA_INDEX[9:8] · SE_INDEX[19:16] ·
//   SA_BROADCAST bit29 · INSTANCE_BROADCAST bit30 · SE_BROADCAST bit31.
static const uint32_t GRBM_SA_BROADCAST_WRITES = 0x20000000u;
static const uint32_t GRBM_INSTANCE_BROADCAST  = 0x40000000u;
static const uint32_t GRBM_SE_BROADCAST_WRITES = 0x80000000u;
// GL2C is a GLOBAL block: addressed flat by INSTANCE_INDEX (NOT per-SE). Select
// one instance i while broadcasting SE+SA so the write lands on that instance.
static inline uint32_t GrbmSelectGl2cInstance(uint32_t i) {
    return (i & 0x7fu) | GRBM_SE_BROADCAST_WRITES | GRBM_SA_BROADCAST_WRITES;
}
static inline uint32_t GrbmBroadcastAll() {
    return GRBM_SE_BROADCAST_WRITES | GRBM_SA_BROADCAST_WRITES | GRBM_INSTANCE_BROADCAST;
}

// --- CP_PERFMON_CNTL fields (gc_12_0_0_sh_mask.h) ---
//   PERFMON_STATE[3:0] · SPM_PERFMON_STATE[7:4] · PERFMON_ENABLE_MODE[9:8] ·
//   PERFMON_SAMPLE_ENABLE bit10.
static const uint32_t CP_PERFMON_STATE_DISABLE_RESET = 0u;
static const uint32_t CP_PERFMON_STATE_START         = 1u;
static const uint32_t CP_PERFMON_STATE_STOP          = 2u;
static const uint32_t CP_PERFMON_SAMPLE_ENABLE       = 0x400u;  // bit10
static inline uint32_t CpPerfmonCntl(uint32_t state, bool sample) {
    return (state & 0xfu) | (sample ? CP_PERFMON_SAMPLE_ENABLE : 0u);
}

// --- GL2C_PERFCOUNTERx_SELECT fields (gc_12_0_0_sh_mask.h) ---
//   PERF_SEL[9:0] · PERF_SEL1[19:10] · CNTR_MODE[23:20] · PERF_MODE1[27:24] ·
//   PERF_MODE[31:28]. One event per counter, accumulate mode (PERF_MODE=0).
static inline uint32_t Gl2cSelect(uint32_t perf_sel) { return perf_sel & 0x3ffu; }

// --- GL2C PERF_SEL event ids (gfx1201; rocprofiler counter_defs.yaml) ---
//   NB: gfx12 renumbers vs gfx11 -- e.g. gfx11 RDREQ_32B=99, gfx12=146; and
//   gfx12 MISS=42 == gfx11 HIT=42. Index by name+arch, never raw number.
static const uint32_t GL2C_EVT_EA_RDREQ_32B  = 146u;
static const uint32_t GL2C_EVT_EA_RDREQ_64B  = 147u;
static const uint32_t GL2C_EVT_EA_RDREQ_128B = 148u;
static const uint32_t GL2C_EVT_EA_WRREQ      = 108u;
static const uint32_t GL2C_EVT_HIT           = 41u;
static const uint32_t GL2C_EVT_MISS          = 42u;

// gfx1201 (Navi48) has 32 GL2C instances (PAL gfx12Device + aqlprofile gfx1201).
static const uint32_t NUM_GL2C_INSTANCES = 32u;

// gfx12 FETCH (read) bytes from the three read-size counters. No 96B on gfx12.
static inline uint64_t Gl2cFetchBytes(uint64_t rd32, uint64_t rd64, uint64_t rd128) {
    return rd32*32ull + rd64*64ull + rd128*128ull;
}

// ---------------------------------------------------------------------------
// PM4WriteRegPacket : WRITE_DATA to a single memory-mapped register.
//
//   regDwordOffset : absolute MMIO register offset in DWORDs (e.g. the value
//                    from the gfx12 GL2C/GRBM/CP_PERFMON table).
//   value          : the 32-bit value to write.
//   engineSel      : 0=ME (default; perfcounter programming runs on ME).
//
// Reuses the vendored PM4WRITE_DATA_CI layout but flips dst_sel to
// MEM_MAPPED_REGISTER and puts the register offset in the address field.
// ---------------------------------------------------------------------------
class PM4WriteRegPacket : public PM4Packet {
 public:
    PM4WriteRegPacket(uint32_t regDwordOffset, uint32_t value,
                      unsigned engineSel = engine_sel_write_data_ci_MICRO_ENGINE_0) {
        m_pkt = reinterpret_cast<PM4WRITE_DATA_CI *>(AllocPacket());
        memset(m_pkt, 0, SizeInBytes());

        InitPM4Header(m_pkt->header, IT_WRITE_DATA);

        m_pkt->bitfields2.dst_sel      = dst_sel_mec_write_data_MEM_MAPPED_REGISTER_0;
        m_pkt->bitfields2.addr_incr    = addr_incr_mec_write_data_INCREMENT_ADDR_0;
        m_pkt->bitfields2.wr_confirm   = wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1;
        m_pkt->bitfields2.cache_policy = cache_policy_mec_write_data_BYPASS_2;
        m_pkt->bitfields2.engine_sel   = (WRITE_DATA_CI_engine_sel)engineSel;

        // For MEM_MAPPED_REGISTER the address field carries the register DWORD
        // offset directly; high dword is 0.
        m_pkt->dst_addr_lo    = regDwordOffset;
        m_pkt->dst_address_hi = 0;
        m_pkt->data[0]        = value;
    }
    virtual ~PM4WriteRegPacket(void) {}

    // header + ordinal2 + dst_lo + dst_hi + 1 data dword = 5 dwords.
    virtual unsigned int SizeInBytes() const {
        return offsetof(PM4WRITE_DATA_CI, data) + sizeof(uint32_t);
    }
    virtual const void *GetPacket() const { return m_pkt; }

 private:
    PM4WRITE_DATA_CI *m_pkt;
};

// ---------------------------------------------------------------------------
// PM4CopyDataPacket : COPY_DATA, hand-encoded as 6 raw dwords.
//
//   srcSel      : COPY_DATA_SRC_* (PERFCOUNTERS or MEM_MAPPED_REG for a
//                 perfcounter LO/HI read).
//   srcRegOrLo  : for register/perfcounter src, the source register DWORD
//                 offset; for memory src, the low 32 bits of the source VA.
//   srcHi       : high 32 bits of a memory source VA (0 for register src).
//   dstGpuVa    : destination GPU virtual address (our CPU-visible result BO).
//   count64     : false => copy 32 bits (one counter dword);
//                 true  => copy 64 bits (LO+HI as a unit, if HW supports it).
//   engineSel   : 0=ME.
//
// Layout (dwords): [0]=header  [1]=control  [2]=src_lo/reg  [3]=src_hi
//                  [4]=dst_lo  [5]=dst_hi.
// ---------------------------------------------------------------------------
class PM4CopyDataPacket : public PM4Packet {
 public:
    PM4CopyDataPacket(unsigned srcSel, uint32_t srcRegOrLo, uint32_t srcHi,
                      uint64_t dstGpuVa, bool count64 = false,
                      unsigned dstSel = COPY_DATA_DST_MEMORY,
                      unsigned engineSel = COPY_DATA_ENGINE_ME) {
        m_dw = reinterpret_cast<uint32_t *>(AllocPacket());
        memset(m_dw, 0, SizeInBytes());

        // Type-3 header via the base helper (writes m_dw[0]); it reads
        // SizeInDWords() to compute the count field.
        InitPM4Header(*reinterpret_cast<PM4_TYPE_3_HEADER *>(&m_dw[0]), IT_COPY_DATA);

        uint32_t control = 0;
        control |= (srcSel    & 0xf) << 0;
        control |= (dstSel    & 0xf) << 8;
        control |= (uint32_t)(count64 ? COPY_DATA_COUNT_64BIT : COPY_DATA_COUNT_32BIT) << 16;
        control |= 1u << 20;                       // wr_confirm
        control |= (engineSel & 0x3) << 30;
        m_dw[1] = control;

        m_dw[2] = srcRegOrLo;                      // src reg dword offset, or VA lo
        m_dw[3] = srcHi;                           // src VA hi (0 for register src)
        m_dw[4] = (uint32_t)(dstGpuVa & 0xffffffffull);
        m_dw[5] = (uint32_t)(dstGpuVa >> 32);
    }
    virtual ~PM4CopyDataPacket(void) {}

    virtual unsigned int SizeInBytes() const { return 6 * sizeof(uint32_t); }
    virtual const void *GetPacket() const { return m_dw; }

 private:
    uint32_t *m_dw;
};

#endif  // RDNA4_DVGPR_PM4_PERF_H
