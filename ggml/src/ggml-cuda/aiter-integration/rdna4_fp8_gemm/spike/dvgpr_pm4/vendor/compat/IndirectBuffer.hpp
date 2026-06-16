// compat/IndirectBuffer.hpp
//
// Minimal SHIM. The vendored PM4Packet.hpp declares a PM4IndirectBufPacket
// class whose InitPacket() touches IndirectBuffer::Addr()/SizeInDWord(). We
// place all packets DIRECTLY in the compute ring (no indirect buffer), so we
// never construct a PM4IndirectBufPacket -- but the member function is still
// compiled in the vendored translation unit and must link. Inline trivial
// definitions keep the symbol satisfied without dragging in kfdtest's real
// IndirectBuffer (which pulls KFDTestUtil/HsaMemoryBuffer/gtest).
#ifndef __KFD_COMPAT_INDIRECTBUFFER_SHIM__
#define __KFD_COMPAT_INDIRECTBUFFER_SHIM__

class IndirectBuffer {
 public:
    unsigned int *Addr() { return nullptr; }
    unsigned int SizeInDWord() { return 0; }
};

#endif  // __KFD_COMPAT_INDIRECTBUFFER_SHIM__
