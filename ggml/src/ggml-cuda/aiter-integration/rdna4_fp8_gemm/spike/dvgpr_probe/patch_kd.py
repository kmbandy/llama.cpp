#!/usr/bin/env python3
"""
patch_kd.py - set COMPUTE_PGM_RSRC3 bit 17 (ENABLE_DYNAMIC_VGPR) in a HIP code
object's kernel descriptor, producing a patched copy. NO GPU, NO LAUNCH.

The bit being set:
  COMPUTE_PGM_RSRC3.ENABLE_DYNAMIC_VGPR = bit 17 (width 1).
  Defined only for GFX125 (gfx1250). On gfx1201 this region is RESERVED.

Kernel descriptor layout (AMDHSA, verified against
/opt/rocm/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h):
  COMPUTE_PGM_RSRC3 is at byte offset 44 inside the 64-byte kernel_descriptor_t
  (COMPUTE_PGM_RSRC3_OFFSET = 44). NOT 12 - byte 12 is reserved0.

A hipcc --genco .hsaco is a __CLANG_OFFLOAD_BUNDLE__ wrapping the amdgcn ELF.
We locate the gfx1201 bundle entry, parse the ELF inside it to find the
'<kernel>.kd' symbol, map its st_value -> file offset via the containing
section, then flip bit 17 of the uint32 at (kd_file_offset + 44).
"""

import struct
import sys
import shutil

KD_SIZE = 64
COMPUTE_PGM_RSRC3_OFFSET = 44   # verified against AMDHSAKernelDescriptor.h
ENABLE_DYNAMIC_VGPR_BIT = 17

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def find_gfx_bundle_entry(data, want_substr=b"amdhsa--gfx"):
    """Return (entry_offset, entry_size) of the first amdgcn/amdhsa code object
    entry inside a clang offload bundle, or (0, len) if the file is a raw ELF."""
    if data[:4] == b"\x7fELF":
        return 0, len(data)
    if not data.startswith(BUNDLE_MAGIC):
        raise SystemExit("error: input is neither a raw ELF nor a clang offload bundle")
    # Header: magic(24) | uint64 num_entries | then per entry:
    #   uint64 offset | uint64 size | uint64 id_len | char id[id_len]
    pos = len(BUNDLE_MAGIC)
    (num_entries,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    for _ in range(num_entries):
        offset, size, id_len = struct.unpack_from("<QQQ", data, pos)
        pos += 24
        entry_id = data[pos:pos + id_len]
        pos += id_len
        if want_substr in entry_id:
            return offset, size
    raise SystemExit("error: no gfx code-object entry found in bundle")


def find_kd_symbol(elf):
    """Parse a little-endian ELF64; return (st_value, st_shndx) for a *.kd symbol."""
    assert elf[:4] == b"\x7fELF", "not an ELF"
    assert elf[4] == 2, "expected ELF64"
    assert elf[5] == 1, "expected little-endian"
    e_shoff, = struct.unpack_from("<Q", elf, 0x28)
    e_shentsize, = struct.unpack_from("<H", elf, 0x3A)
    e_shnum, = struct.unpack_from("<H", elf, 0x3C)
    e_shstrndx, = struct.unpack_from("<H", elf, 0x3E)

    sections = []  # (name_off, type, addr, offset, size, link, entsize)
    for i in range(e_shnum):
        b = e_shoff + i * e_shentsize
        sh_name, sh_type = struct.unpack_from("<II", elf, b)
        sh_addr, sh_offset, sh_size = struct.unpack_from("<QQQ", elf, b + 0x10)
        sh_link, = struct.unpack_from("<I", elf, b + 0x28)
        sh_entsize, = struct.unpack_from("<Q", elf, b + 0x38)
        sections.append((sh_name, sh_type, sh_addr, sh_offset, sh_size, sh_link, sh_entsize))

    # Walk symbol tables (SHT_SYMTAB=2, SHT_DYNSYM=11); resolve names via linked strtab.
    for sh_name, sh_type, _, sh_offset, sh_size, sh_link, sh_entsize in sections:
        if sh_type not in (2, 11) or sh_entsize == 0:
            continue
        str_off = sections[sh_link][3]
        for so in range(sh_offset, sh_offset + sh_size, sh_entsize):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", elf, so)
            if st_name == 0:
                continue
            end = elf.index(b"\x00", str_off + st_name)
            name = elf[str_off + st_name:end].decode("utf-8", "replace")
            if name.endswith(".kd"):
                shaddr = sections[st_shndx][2]
                shoff = sections[st_shndx][3]
                # file offset of the .kd within this ELF
                kd_file_off = shoff + (st_value - shaddr)
                return name, st_value, st_shndx, kd_file_off
    raise SystemExit("error: no '*.kd' symbol found in ELF")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "probe.hsaco"
    dst = sys.argv[2] if len(sys.argv) > 2 else "probe_patched.hsaco"

    with open(src, "rb") as f:
        data = bytearray(f.read())

    entry_off, entry_size = find_gfx_bundle_entry(data)
    elf = bytes(data[entry_off:entry_off + entry_size])
    kd_name, st_value, st_shndx, kd_rel_off = find_kd_symbol(elf)

    # absolute file offset of the kernel descriptor in the bundle file
    kd_abs = entry_off + kd_rel_off
    rsrc3_abs = kd_abs + COMPUTE_PGM_RSRC3_OFFSET

    before, = struct.unpack_from("<I", data, rsrc3_abs)
    after = before | (1 << ENABLE_DYNAMIC_VGPR_BIT)

    print(f"input                : {src}")
    print(f"bundle entry offset  : 0x{entry_off:x} (size 0x{entry_size:x})")
    print(f"kd symbol            : {kd_name}  st_value=0x{st_value:x} shndx={st_shndx}")
    print(f"kd file offset (abs) : 0x{kd_abs:x}")
    print(f"compute_pgm_rsrc3 @  : 0x{rsrc3_abs:x}  (= kd 0x{kd_abs:x} + {COMPUTE_PGM_RSRC3_OFFSET})")
    print(f"compute_pgm_rsrc3 BEFORE : 0x{before:08x}")
    print(f"compute_pgm_rsrc3 AFTER  : 0x{after:08x}   (bit {ENABLE_DYNAMIC_VGPR_BIT} set)")

    if before == after:
        print("WARNING: bit 17 already set in source; output will be identical.")

    struct.pack_into("<I", data, rsrc3_abs, after)

    # write the patched copy
    with open(dst, "wb") as f:
        f.write(data)

    # sanity: diff source vs patched - must differ in exactly one byte/bit
    with open(src, "rb") as f:
        orig = f.read()
    with open(dst, "rb") as f:
        patched = f.read()
    assert len(orig) == len(patched), "size changed!"
    diffs = [(i, orig[i], patched[i]) for i in range(len(orig)) if orig[i] != patched[i]]
    print(f"\nbyte diff (src vs patched): {len(diffs)} byte(s)")
    for i, a, b in diffs:
        xor = a ^ b
        bits = [k for k in range(8) if (xor >> k) & 1]
        print(f"  offset 0x{i:x}: 0x{a:02x} -> 0x{b:02x}  (changed bit(s) in this byte: {bits})")
    # bit 17 of the uint32 -> byte (44+2)=offset rsrc3_abs+2, intra-byte bit 1
    expected_byte = rsrc3_abs + (ENABLE_DYNAMIC_VGPR_BIT // 8)
    expected_intrabyte_bit = ENABLE_DYNAMIC_VGPR_BIT % 8
    ok = (len(diffs) == 1
          and diffs[0][0] == expected_byte
          and (diffs[0][1] ^ diffs[0][2]) == (1 << expected_intrabyte_bit))
    print(f"expected single-bit change at byte 0x{expected_byte:x}, intra-byte bit "
          f"{expected_intrabyte_bit}: {'OK' if ok else 'MISMATCH'}")
    print(f"\nwrote                : {dst}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
