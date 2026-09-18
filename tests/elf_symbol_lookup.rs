//! Function symbol lookup by name and by address on a parsed ELF.
//!
//! The image is assembled by hand, so the test needs no toolchain: an ELF64
//! header, one executable `PT_LOAD` covering the whole file, and a section
//! table carrying a `.symtab` / `.strtab` pair.

use wazabin_binary::BinaryFormat;
use wazabin_binary::elf::ElfBinary;

const EHDR: usize = 64;
const PHENT: usize = 56;
const SHENT: usize = 64;
const SYMENT: usize = 24;
const BASE: u64 = 0x400000;
const ENTRY: u64 = BASE + 0x1000;

const PT_LOAD: u32 = 1;
const PF_X: u32 = 1;
const PF_R: u32 = 4;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const STB_GLOBAL: u8 = 1;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

struct Sym {
    name: &'static str,
    kind: u8,
    value: u64,
}

/// An `ET_EXEC` image whose `.symtab` holds `syms` (plus the null symbol).
fn image(e_type: u16, syms: &[Sym]) -> Vec<u8> {
    // String table: a leading NUL, then each name NUL-terminated.
    let mut strtab = vec![0u8];
    let mut name_offsets = Vec::new();
    for sym in syms {
        name_offsets.push(strtab.len() as u32);
        strtab.extend_from_slice(sym.name.as_bytes());
        strtab.push(0);
    }
    let shstrtab = b"\0.symtab\0.strtab\0.shstrtab\0".to_vec();

    let symtab_off = EHDR + PHENT;
    let symtab_len = SYMENT * (syms.len() + 1);
    let strtab_off = symtab_off + symtab_len;
    let shstrtab_off = strtab_off + strtab.len();
    let shoff = (shstrtab_off + shstrtab.len()).next_multiple_of(8);
    let file_len = shoff + SHENT * 4;

    let mut out = vec![0u8; file_len];
    out[..4].copy_from_slice(b"\x7fELF");
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // little-endian
    out[6] = 1; // EV_CURRENT
    out[16..18].copy_from_slice(&e_type.to_le_bytes());
    out[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[24..32].copy_from_slice(&ENTRY.to_le_bytes());
    out[32..40].copy_from_slice(&(EHDR as u64).to_le_bytes()); // e_phoff
    out[40..48].copy_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
    out[52..54].copy_from_slice(&(EHDR as u16).to_le_bytes());
    out[54..56].copy_from_slice(&(PHENT as u16).to_le_bytes());
    out[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
    out[58..60].copy_from_slice(&(SHENT as u16).to_le_bytes());
    out[60..62].copy_from_slice(&4u16.to_le_bytes()); // e_shnum
    out[62..64].copy_from_slice(&3u16.to_le_bytes()); // e_shstrndx

    // One R+X PT_LOAD at file offset 0 covering the file, with a zero-filled
    // tail reaching past the entrypoint and every symbol below.
    let ph = &mut out[EHDR..EHDR + PHENT];
    ph[0..4].copy_from_slice(&PT_LOAD.to_le_bytes());
    ph[4..8].copy_from_slice(&(PF_R | PF_X).to_le_bytes());
    ph[16..24].copy_from_slice(&BASE.to_le_bytes());
    ph[24..32].copy_from_slice(&BASE.to_le_bytes());
    ph[32..40].copy_from_slice(&(file_len as u64).to_le_bytes());
    ph[40..48].copy_from_slice(&0x3000u64.to_le_bytes());
    ph[48..56].copy_from_slice(&0x1000u64.to_le_bytes());

    for (i, sym) in syms.iter().enumerate() {
        let at = symtab_off + SYMENT * (i + 1);
        let entry = &mut out[at..at + SYMENT];
        entry[0..4].copy_from_slice(&name_offsets[i].to_le_bytes());
        entry[4] = (STB_GLOBAL << 4) | sym.kind;
        entry[6..8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx: any defined section
        entry[8..16].copy_from_slice(&sym.value.to_le_bytes());
    }
    out[strtab_off..strtab_off + strtab.len()].copy_from_slice(&strtab);
    out[shstrtab_off..shstrtab_off + shstrtab.len()].copy_from_slice(&shstrtab);

    // Section headers: null, .symtab, .strtab, .shstrtab.
    let mut shdr =
        |idx: usize, name: u32, kind: u32, off: usize, size: usize, link: u32, entsize: u64| {
            let at = shoff + SHENT * idx;
            let sh = &mut out[at..at + SHENT];
            sh[0..4].copy_from_slice(&name.to_le_bytes());
            sh[4..8].copy_from_slice(&kind.to_le_bytes());
            sh[24..32].copy_from_slice(&(off as u64).to_le_bytes());
            sh[32..40].copy_from_slice(&(size as u64).to_le_bytes());
            sh[40..44].copy_from_slice(&link.to_le_bytes());
            sh[44..48].copy_from_slice(&1u32.to_le_bytes()); // sh_info: first non-local
            sh[48..56].copy_from_slice(&1u64.to_le_bytes()); // sh_addralign
            sh[56..64].copy_from_slice(&entsize.to_le_bytes());
        };
    shdr(1, 1, SHT_SYMTAB, symtab_off, symtab_len, 2, SYMENT as u64);
    shdr(2, 9, SHT_STRTAB, strtab_off, strtab.len(), 0, 0);
    shdr(3, 17, SHT_STRTAB, shstrtab_off, shstrtab.len(), 0, 0);
    out
}

fn parse(e_type: u16, syms: &[Sym]) -> ElfBinary {
    ElfBinary::parse(&image(e_type, syms)).expect("hand-built ELF parses")
}

const SYMS: &[Sym] = &[
    Sym {
        name: "main",
        kind: STT_FUNC,
        value: BASE + 0x1136,
    },
    Sym {
        name: "helper",
        kind: STT_FUNC,
        value: BASE + 0x1200,
    },
    // A second `static helper` from another translation unit, at a lower
    // address than the first so the tie-break is observable.
    Sym {
        name: "helper",
        kind: STT_FUNC,
        value: BASE + 0x1100,
    },
    // Not a function: must not answer to a function lookup.
    Sym {
        name: "table",
        kind: STT_OBJECT,
        value: BASE + 0x2000,
    },
];

#[test]
fn looks_up_by_name_and_by_address() {
    let elf = parse(2, SYMS); // ET_EXEC

    assert_eq!(elf.symbol_address("main"), Some(0x401136));
    assert_eq!(elf.symbol_name(0x401136), Some("main"));
    assert_eq!(elf.symbol_name(0x401137), None);
    assert_eq!(elf.symbol_address("missing"), None);
    assert_eq!(elf.symbol_address("table"), None);
    assert_eq!(elf.symbol_name(0x402000), None);
}

#[test]
fn duplicate_names_resolve_to_the_lowest_address() {
    let elf = parse(2, SYMS);

    assert_eq!(elf.symbol_address("helper"), Some(0x401100));
    assert_eq!(elf.symbol_name(0x401100), Some("helper"));
    assert_eq!(elf.symbol_name(0x401200), Some("helper"));
}

#[test]
fn unnamed_entrypoint_reads_as_start_both_ways() {
    let elf = parse(2, SYMS);

    assert_eq!(elf.symbol_name(ENTRY), Some("_start"));
    assert_eq!(elf.symbol_address("_start"), Some(ENTRY));

    // Once a real symbol names the entrypoint, `_start` is no longer minted.
    let named = parse(
        2,
        &[Sym {
            name: "start_here",
            kind: STT_FUNC,
            value: ENTRY,
        }],
    );
    assert_eq!(named.symbol_name(ENTRY), Some("start_here"));
    assert_eq!(named.symbol_address("start_here"), Some(ENTRY));
    assert_eq!(named.symbol_address("_start"), None);
}

/// The lookups hand back link-time addresses; for a PIE that is the offset
/// from whatever base the loader chooses, the same convention as
/// `load_address`.
#[test]
fn pie_addresses_are_link_relative() {
    let elf = parse(3, SYMS); // ET_DYN

    assert_eq!(elf.load_address(), BASE);
    assert_eq!(elf.symbol_address("main"), Some(BASE + 0x1136));

    // A loader that maps the image at `runtime_base` reaches `main` at
    // `runtime_base + (link address - load_address)`, and must undo the same
    // shift before asking for a name.
    let runtime_base = 0x5555_5555_4000u64;
    let main_at_runtime = runtime_base + (BASE + 0x1136 - elf.load_address());
    assert_eq!(elf.symbol_name(main_at_runtime), None);
    assert_eq!(
        elf.symbol_name(main_at_runtime - runtime_base + elf.load_address()),
        Some("main")
    );
}

#[test]
fn every_known_function_round_trips() {
    let elf = parse(2, SYMS);

    for f in &elf.analysis.known_functions {
        let Some(name) = f.name.as_deref() else {
            continue;
        };
        assert_eq!(elf.symbol_name(f.address), Some(name));
        let addr = elf.symbol_address(name).expect("indexed");
        assert_eq!(elf.symbol_name(addr), Some(name));
    }
}
