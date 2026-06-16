// vendor/BasePacket.cpp
//
// Drop-in reimplementation of kfdtest's BasePacket.cpp that removes the gtest /
// g_baseTest dependency. Behaviour is identical to upstream (ba56a24c) for the
// paths PM4Packet uses:
//   * AllocPacket() = calloc(1, SizeInBytes())  (upstream uses calloc too)
//   * m_FamilyId default = 0 (FAMILY_UNKNOWN); every PM4 packet that branches on
//     family sets m_FamilyId itself from the familyId ctor arg, so the default
//     is never consulted on the encode paths we exercise.
//   * Dump() is a no-op (debug only).
#include "BasePacket.hpp"
#include <cstdlib>

BasePacket::BasePacket(void) : m_FamilyId(0), m_packetAllocation(NULL) {}

BasePacket::~BasePacket(void) {
    if (m_packetAllocation)
        free(m_packetAllocation);
}

void BasePacket::Dump() const {}

void *BasePacket::AllocPacket(void) {
    unsigned int size = SizeInBytes();
    if (!size)
        return NULL;
    m_packetAllocation = calloc(1, size);
    return m_packetAllocation;
}
