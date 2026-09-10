//! The fields a loader needs beyond the segments: the file type, where the
//! program header table sits, and the thread-local storage template.
//!
//! The images are assembled here byte by byte, so the test needs no
//! toolchain: an ELF64 header followed by three program headers, the first
//! `PT_LOAD` covering the headers themselves as a linker would lay it out.

use wazabin_binary::elf::{ElfBinary, ElfKind};

const EHDR: usize = 64;
const PHENT: usize = 56;
const PT_LOAD: u32 = 1;
const PT_PHDR: u32 = 6;
const PT_TLS: u32 = 7;

struct Phdr {
    kind: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

fn image(e_type: u16, phdrs: &[Phdr]) -> Vec<u8> {
    let mut out = vec![0u8; EHDR + PHENT * phdrs.len()];
    out[..4].copy_from_slice(b"\x7fELF");
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // little-endian
    out[6] = 1; // EV_CURRENT
    out[16..18].copy_from_slice(&e_type.to_le_bytes());
    out[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[24..32].copy_from_slice(&0x1000u64.to_le_bytes()); // e_entry
    out[32..40].copy_from_slice(&(EHDR as u64).to_le_bytes()); // e_phoff
    out[52..54].copy_from_slice(&(EHDR as u16).to_le_bytes());
    out[54..56].copy_from_slice(&(PHENT as u16).to_le_bytes());
    out[56..58].copy_from_slice(&(phdrs.len() as u16).to_le_bytes());
    out[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    for (i, p) in phdrs.iter().enumerate() {
        let at = EHDR + i * PHENT;
        let ph = &mut out[at..at + PHENT];
        ph[0..4].copy_from_slice(&p.kind.to_le_bytes());
        ph[4..8].copy_from_slice(&p.flags.to_le_bytes());
        ph[8..16].copy_from_slice(&p.offset.to_le_bytes());
        ph[16..24].copy_from_slice(&p.vaddr.to_le_bytes());
        ph[24..32].copy_from_slice(&p.vaddr.to_le_bytes());
        ph[32..40].copy_from_slice(&p.filesz.to_le_bytes());
        ph[40..48].copy_from_slice(&p.memsz.to_le_bytes());
        ph[48..56].copy_from_slice(&p.align.to_le_bytes());
    }
    // Pad so the PT_LOAD's file bytes exist; nothing reads them.
    out.resize(0x200, 0);
    out
}

/// A load segment at file offset 0 covering the headers, a TLS template
/// inside it, and no `PT_PHDR`: the table's address is derived from the
/// segment that covers its file offset.
fn without_phdr(e_type: u16, base: u64) -> Vec<u8> {
    image(
        e_type,
        &[
            Phdr {
                kind: PT_LOAD,
                flags: 5,
                offset: 0,
                vaddr: base,
                filesz: 0x200,
                memsz: 0x300,
                align: 0x1000,
            },
            Phdr {
                kind: PT_TLS,
                flags: 4,
                offset: 0x100,
                vaddr: base + 0x100,
                filesz: 0x20,
                memsz: 0x60,
                align: 8,
            },
        ],
    )
}

#[test]
fn executable_without_pt_phdr_locates_the_table_inside_the_load_segment() {
    let elf = ElfBinary::parse(&without_phdr(2, 0x40_0000)).unwrap();
    assert_eq!(elf.kind, ElfKind::Executable);
    assert_eq!(elf.program_headers.offset, 64);
    assert_eq!(elf.program_headers.entry_size, 56);
    assert_eq!(elf.program_headers.count, 2);
    assert_eq!(elf.program_headers.vaddr, Some(0x40_0040));
    let tls = elf.tls.expect("PT_TLS is reported");
    assert_eq!(tls.vaddr, 0x40_0100);
    assert_eq!(tls.file_size, 0x20);
    assert_eq!(tls.mem_size, 0x60);
    assert_eq!(tls.align, 8);
    assert_eq!(elf.segments.len(), 1, "PT_TLS is not a load segment");
    assert_eq!(elf.segments[0].align, 0x1000);
}

#[test]
fn pie_is_a_shared_object_with_unrelocated_addresses() {
    let elf = ElfBinary::parse(&without_phdr(3, 0)).unwrap();
    assert_eq!(elf.kind, ElfKind::SharedObject);
    assert_eq!(elf.program_headers.vaddr, Some(0x40));
    assert_eq!(elf.tls.unwrap().vaddr, 0x100);
}

#[test]
fn pt_phdr_names_the_table_outright() {
    let bytes = image(
        2,
        &[
            Phdr {
                kind: PT_PHDR,
                flags: 4,
                offset: 64,
                vaddr: 0x40_0040,
                filesz: 56 * 2,
                memsz: 56 * 2,
                align: 8,
            },
            // The load segment starts past the headers, so without PT_PHDR
            // the table would have no address at all.
            Phdr {
                kind: PT_LOAD,
                flags: 5,
                offset: 0x100,
                vaddr: 0x40_1000,
                filesz: 0x100,
                memsz: 0x100,
                align: 0x1000,
            },
        ],
    );
    let elf = ElfBinary::parse(&bytes).unwrap();
    assert_eq!(elf.program_headers.vaddr, Some(0x40_0040));
    assert_eq!(elf.tls, None);
}

#[test]
fn a_table_no_segment_covers_has_no_address() {
    let bytes = image(
        2,
        &[Phdr {
            kind: PT_LOAD,
            flags: 5,
            offset: 0x100,
            vaddr: 0x40_1000,
            filesz: 0x100,
            memsz: 0x100,
            align: 0x1000,
        }],
    );
    let elf = ElfBinary::parse(&bytes).unwrap();
    assert_eq!(elf.program_headers.vaddr, None);
}
