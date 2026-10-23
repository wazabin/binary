//! The borrowed loader view on a byte-assembled PIE: segments, sections,
//! relocations, symbols, imports (all three PLT stub kinds), the dynamic
//! table, TLS, and the section-less reading.

mod common;

use common::*;
use wazabin_binary::image::{Image, RelocKind, StubKind, SymbolKind, Table};
use wazabin_binary::{LoadError, elf::ElfKind};

#[test]
fn segments_carry_flags_offsets_and_file_bytes() {
    let bytes = build(true);
    let img = Image::load(&bytes).unwrap();
    assert_eq!(img.kind, ElfKind::SharedObject);
    assert!(img.is_pie() && img.is_64());
    assert_eq!(img.machine, 62);
    assert_eq!(img.entry, 0x700);
    assert_eq!(img.segments.len(), 2);
    let (text, data) = (&img.segments[0], &img.segments[1]);
    assert_eq!(
        (text.vaddr, text.file_off, text.memsz, text.flags),
        (0, 0, 0xa00, 5)
    );
    assert!(text.executable() && !text.writable());
    assert_eq!(text.bytes.len(), 0xa00);
    assert_eq!(
        (data.vaddr, data.file_off, data.memsz, data.align),
        (0x3000, 0x1000, 0x600, 0x1000)
    );
    assert!(data.writable() && !data.executable());
    assert_eq!(data.bytes.len(), 0x408);
    assert_eq!(img.read(0x700, 2), Some(&[0xc3, 0xc3][..]));
    assert_eq!(img.bytes_from(0x3400).unwrap().len(), 8);
    assert_eq!(
        img.read(0x3400, 9),
        None,
        "the file-backed bytes end at the template"
    );
    assert_eq!(img.relro, Some((0x3000, 0x200)));
}

#[test]
fn sections_are_the_allocated_headers_and_empty_without_them() {
    let bytes = build(true);
    let img = Image::load(&bytes).unwrap();
    let names: Vec<_> = img.sections.iter().map(|s| s.name).collect();
    assert_eq!(
        names,
        [
            ".dynsym",
            ".dynstr",
            ".gnu.version",
            ".rela.dyn",
            ".rela.plt",
            ".plt",
            ".plt.got",
            ".plt.sec",
            ".text",
            ".gcc_except_table",
            ".eh_frame_hdr",
            ".eh_frame",
            ".dynamic",
            ".init_array",
            ".tbss"
        ]
    );
    let text = img.section(".text").unwrap();
    assert_eq!(
        (
            text.addr,
            text.size,
            text.nobits,
            text.executable(),
            text.writable()
        ),
        (0x700, 0x60, false, true, false)
    );
    assert_eq!(text.bytes.len(), 0x60);
    let tbss = img.section(".tbss").unwrap();
    assert!(tbss.nobits && tbss.tls() && tbss.bytes.is_empty());

    // The same file read through the program headers alone: no names invented.
    let seg = Image::load_segments(&bytes).unwrap();
    assert!(seg.sections.is_empty());
    assert!(seg.symtab().is_empty());
    // Stripped of its section headers altogether.
    let mut stripped = bytes.clone();
    stripped[40..48].fill(0);
    stripped[60..64].fill(0);
    assert!(Image::load(&stripped).unwrap().sections.is_empty());
}

#[test]
fn relocations_in_table_order_with_their_kinds() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let got: Vec<_> = img
        .relocs
        .iter()
        .map(|r| (r.offset, r.kind, r.sym, r.addend))
        .collect();
    assert_eq!(
        got,
        [
            (INIT_ARRAY, RelocKind::Relative, 0, 0x710),
            (FINALIZE_SLOT, RelocKind::GlobDat, 2, 0),
            (TYPEINFO_SLOT, RelocKind::GlobDat, 3, 0),
            (PERSONALITY_SLOT, RelocKind::GlobDat, 7, 0),
            (ABS_SLOT, RelocKind::Abs, 4, 8),
            (IREL_SLOT, RelocKind::IRelative, 0, 0x720),
            (TPOFF_SLOT, RelocKind::TpOff, 0, 0x10),
            (PUTS_SLOT, RelocKind::JumpSlot, 1, 0),
            (FREE_SLOT, RelocKind::JumpSlot, 6, 0),
        ]
    );
    assert_eq!(img.reloc_at(ABS_SLOT).unwrap().addend, 8);
    assert!(img.reloc_at(0x3300 + 1).is_none());
    assert_eq!(
        img.reloc_symbol(img.reloc_at(PUTS_SLOT).unwrap())
            .unwrap()
            .name,
        "puts"
    );
}

/// `DT_RELR`: an address entry, then a bitmap (bits 1 and 3: the first and
/// the third word after it) are three RELATIVEs whose addends are the words in
/// place. Before, the table was not read at all, so a PIE's packed relative
/// relocations (coreutils-9.10-5.fc44's code pointers in `.data.rel.ro`) were
/// no relocations.
#[test]
fn relr_entries_are_relatives_with_the_word_in_place() {
    let mut b = Buf(build(false));
    // The two spare dynamic entries: DT_RELR at 0x3380, DT_RELRSZ 16.
    b.u64(0x1000 + 16 * 19, 36);
    b.u64(0x1000 + 16 * 19 + 8, 0x3380);
    b.u64(0x1000 + 16 * 20, 35);
    b.u64(0x1000 + 16 * 20 + 8, 16);
    b.u64(0x1380, 0x3310);
    b.u64(0x1388, 0b1011);
    for (at, v) in [(0x1310, 0x700u64), (0x1318, 0x708), (0x1328, 0x710)] {
        b.u64(at, v);
    }
    let img = Image::load(&b.0).unwrap();
    let relr: Vec<_> = img
        .relocs
        .iter()
        .filter(|r| r.offset >= 0x3310 && r.offset < 0x3380)
        .map(|r| (r.offset, r.kind, r.sym, r.addend))
        .collect();
    assert_eq!(
        relr,
        [
            (0x3310, RelocKind::Relative, 0, 0x700),
            (0x3318, RelocKind::Relative, 0, 0x708),
            (0x3328, RelocKind::Relative, 0, 0x710),
        ]
    );
    assert!(img.reloc_at(0x3320).is_none());
    assert_eq!(img.word(0x3328), Some(0x710));
}

#[test]
fn words_apply_the_relocation_the_loader_would() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    // RELATIVE: the addend, the file holds 0.
    assert_eq!(img.word(INIT_ARRAY), Some(0x710));
    // No relocation: the file's word.
    assert_eq!(img.word(INIT_ARRAY + 8), Some(0x718));
    // A PIE's own symbol plus addend (main + 8).
    assert_eq!(img.word(ABS_SLOT), Some(0x708));
    // An undefined symbol: the file word (zero).
    assert_eq!(img.word(FINALIZE_SLOT), Some(0));
    assert_eq!(img.word(0x9999_0000), None);
}

#[test]
fn dynamic_table_names_the_constructors_and_libraries() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let d = &img.dynamic;
    assert_eq!(d.needed, ["libc.so.6"]);
    assert_eq!((d.init, d.fini), (Some(0x730), Some(0x740)));
    assert_eq!(d.init_array, Some((INIT_ARRAY, 16)));
    assert_eq!((d.fini_array, d.preinit_array), (None, None));
    assert_eq!(d.pltgot, Some(0x3200));
    assert_eq!(d.table, Some((0x3000, 0x150)));
    assert!(!d.textrel);
    assert_eq!(
        img.array_words(d.init_array.unwrap()),
        Some(vec![0x710, 0x718])
    );
    assert_eq!(img.array_words((0x9999_0000, 8)), None);

    let mut textrel = bytes.clone();
    // DT_FLAGS = DF_TEXTREL in the spare terminator slot's place.
    textrel[0x1000 + 16 * 19..0x1000 + 16 * 19 + 8].copy_from_slice(&30u64.to_le_bytes());
    textrel[0x1000 + 16 * 19 + 8..0x1000 + 16 * 19 + 16].copy_from_slice(&4u64.to_le_bytes());
    assert!(Image::load(&textrel).unwrap().dynamic.textrel);
}

#[test]
fn symbols_come_from_both_tables_and_agree_without_section_headers() {
    let bytes = build(true);
    let img = Image::load(&bytes).unwrap();
    let dynsyms = img.dynsyms();
    assert_eq!(dynsyms.len(), 8, "the null symbol and seven more");
    let main = &dynsyms[4];
    assert_eq!(
        (
            main.name,
            main.addr,
            main.size,
            main.kind,
            main.defined,
            main.table
        ),
        ("main", 0x700, 0x20, SymbolKind::Func, true, Table::Dynsym)
    );
    assert_eq!(dynsyms[3].kind, SymbolKind::Object);
    assert!(!dynsyms[1].defined);
    let local = img.symtab();
    assert_eq!(local.len(), 2);
    assert_eq!(
        (local[1].name, local[1].addr, local[1].table),
        ("local_fn", 0x710, Table::Symtab)
    );

    // Through the dynamic table alone, `DT_HASH` sizes the same `.dynsym`.
    let seg = Image::load_segments(&bytes).unwrap();
    assert_eq!(seg.dynsyms(), dynsyms);
}

#[test]
fn imports_find_all_three_stub_kinds_and_the_canonical_entry() {
    for sections in [true, false] {
        let bytes = build(sections);
        let img = Image::load(&bytes).unwrap();
        let find = |name: &str| {
            img.imports
                .iter()
                .find(|i| i.name == name)
                .unwrap_or_else(|| panic!("no import {name}"))
        };
        // `.plt`: 16-byte stub, JUMP_SLOT.
        assert_eq!(
            (find("puts").slot, find("puts").stub),
            (Some(PUTS_SLOT), Some(0x610))
        );
        // `.plt.sec`: endbr64, bnd jmp.
        assert_eq!(
            (find("free").slot, find("free").stub),
            (Some(FREE_SLOT), Some(0x650))
        );
        // `.plt.got`: 8-byte stub, nop-padded, GLOB_DAT.
        assert_eq!(
            (find("__cxa_finalize").slot, find("__cxa_finalize").stub),
            (Some(FINALIZE_SLOT), Some(0x640))
        );
        // A GOT slot with no stub.
        assert_eq!(
            (find("_ZTIi").slot, find("_ZTIi").stub),
            (Some(TYPEINFO_SLOT), None)
        );
        // An address-taken import: its canonical PLT entry is the dynsym value.
        assert_eq!(
            (find("qsort").slot, find("qsort").stub),
            (None, Some(0x660))
        );
        assert_eq!(find("puts").stub_kind, Some(StubKind::Plt));
        assert_eq!(find("free").stub_kind, Some(StubKind::Plt));
        assert_eq!(find("__cxa_finalize").stub_kind, Some(StubKind::PltGot));
        assert_eq!(find("qsort").stub_kind, None);
        assert_eq!(find("puts").library, Some("libc.so.6"));
        assert_eq!(find("free").library, Some("libc.so.6"));
        assert_eq!(
            find("_ZTIi").library,
            None,
            "version 1 is the base, not a requirement"
        );

        let puts: Vec<_> = img
            .imports
            .iter()
            .filter(|i| i.name == "puts")
            .map(|i| i.stub)
            .collect();
        assert_eq!(
            puts,
            [Some(0x610), Some(0x670)],
            "one import per stub through the slot"
        );
        assert_eq!(img.import_name(0x670), Some("puts"));
        assert_eq!(img.import_name(0x610), Some("puts"));
        assert_eq!(img.import_name(PUTS_SLOT), Some("puts"));
        assert_eq!(img.import_name(0x650), Some("free"));
        assert_eq!(img.import_name(0x660), Some("qsort"));
        assert_eq!(img.import_name(0x700), None);
        // The `.plt` header entry and the lazy-binding stub tail are not imports.
        assert_eq!(img.import_name(0x600), None);
    }
}

#[test]
fn tls_template_and_flavours() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let tls = img.tls.unwrap();
    assert_eq!((tls.vaddr, tls.memsz, tls.align), (0x3400, 0x20, 8));
    assert_eq!(tls.init, [1, 2, 3, 4, 5, 6, 7, 8]);

    assert!(matches!(
        Image::load(b"MZ\x90\0"),
        Err(LoadError::UnknownFormat)
    ));
    let mut big_endian = bytes.clone();
    big_endian[5] = 2;
    assert!(matches!(Image::load(&big_endian), Err(LoadError::Elf(_))));
    // A segment that runs off the end of the file.
    let short = &bytes[..0x800];
    assert!(matches!(Image::load(short), Err(LoadError::Elf(_))));
}

#[test]
fn metadata_ranges_cover_the_loaders_tables_only() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let meta = img.metadata();
    let covered = |a: u64| meta.iter().any(|&(lo, hi)| a >= lo && a < hi);
    assert!(covered(0), "the ELF header");
    assert!(covered(0x200) && covered(0x300) && covered(0x420) && covered(0x500));
    assert!(!covered(0x700), "text is not metadata");
    // The tables apart: `.dynsym` and `.dynstr` abut and merge, but each ends
    // where it ends.
    let tables = img.loader_tables();
    assert!(tables.contains(&(0x200, 0x300)), "{tables:x?}");
    assert!(tables.windows(2).all(|w| w[0].0 <= w[1].0));
}

#[test]
fn headers_are_the_elf_header_table_and_notes() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    // The six program headers follow the 64-byte ELF header; no PT_NOTE here.
    assert_eq!(img.headers(), [(0, 64 + 6 * 56)]);
    // A note segment is a header range too.
    let mut with_note = bytes.clone();
    let ph = 64 + 56 * 5; // reuse PT_GNU_RELRO's slot
    with_note[ph..ph + 4].copy_from_slice(&4u32.to_le_bytes());
    with_note[ph + 16..ph + 24].copy_from_slice(&0x3c0u64.to_le_bytes());
    with_note[ph + 40..ph + 48].copy_from_slice(&0x20u64.to_le_bytes());
    assert_eq!(
        Image::load(&with_note).unwrap().headers(),
        [(0, 400), (0x3c0, 0x3e0)]
    );
    // Two notes that abut are one range.
    let mut two = with_note.clone();
    let ph = 64 + 56 * 3; // PT_TLS's slot
    two[ph..ph + 4].copy_from_slice(&4u32.to_le_bytes());
    two[ph + 16..ph + 24].copy_from_slice(&0x3e0u64.to_le_bytes());
    two[ph + 40..ph + 48].copy_from_slice(&0x10u64.to_le_bytes());
    two[ph + 8..ph + 16].copy_from_slice(&0x3e0u64.to_le_bytes());
    assert_eq!(
        Image::load(&two).unwrap().headers(),
        [(0, 400), (0x3c0, 0x3f0)]
    );
}

#[test]
fn gnu_hash_symbol_count_reads_one_chain() {
    // 100 000 buckets that all start one unterminated 100 000-entry chain:
    // walking every bucket's chain is 10^10 steps, the last chain alone 10^5.
    let n = 100_000;
    let bytes = with_gnu_hash(1, &vec![1; n], &vec![0; n]);
    let (done, wait) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let img = Image::load_segments(&bytes).unwrap();
        done.send(img.dynsyms().len()).unwrap();
    });
    let count = wait
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the symbol count is linear in the hash table");
    // As many as `.dynsym` holds before its bytes run out.
    assert!(count > 8, "{count}");
}

#[test]
fn an_import_only_gnu_hash_counts_the_symbols_the_relocations_name() {
    // An executable that only imports: `symoffset` 1 and no bucket chains a
    // symbol. Its seven imports follow the offset, and the relocations name
    // them all.
    let bytes = with_gnu_hash(1, &[0], &[]);
    let img = Image::load_segments(&bytes).unwrap();
    let full = build(false);
    let with_hash = Image::load_segments(&full).unwrap();
    assert_eq!(img.dynsyms().len(), 8);
    assert_eq!(img.dynsyms(), with_hash.dynsyms());
    for r in &img.relocs {
        assert_eq!(
            img.reloc_symbol(r).map(|s| s.name),
            with_hash.reloc_symbol(r).map(|s| s.name),
            "relocation at {:#x}",
            r.offset
        );
    }
    assert_eq!(img.imports, with_hash.imports);
}

/// An x32 executable (ELF32, x86-64): two `JUMP_SLOT`s in a `DT_JMPREL`
/// table that `DT_PLTREL` says is RELA.
fn x32() -> Vec<u8> {
    let mut b = Buf(vec![0u8; 0x400]);
    b.put(0, b"\x7fELF");
    b.put(4, &[1, 1, 1]);
    b.u16(16, 2); // ET_EXEC
    b.u16(18, 62);
    b.u32(20, 1);
    b.u32(28, 52); // e_phoff
    b.u16(40, 52);
    b.u16(42, 32);
    b.u16(44, 2);
    // PT_LOAD (RW) over the whole file, PT_DYNAMIC at 0x300.
    for (i, (kind, off, size, flags)) in [(1u32, 0u32, 0x400u32, 6u32), (2, 0x300, 0x40, 6)]
        .into_iter()
        .enumerate()
    {
        let at = 52 + 32 * i;
        b.u32(at, kind);
        b.u32(at + 4, off);
        b.u32(at + 8, off);
        b.u32(at + 12, off);
        b.u32(at + 16, size);
        b.u32(at + 20, size);
        b.u32(at + 24, flags);
        b.u32(at + 28, 4);
    }
    // .dynsym: null, f, g (global FUNC, undefined); .dynstr.
    for (i, name) in [1u32, 3].into_iter().enumerate() {
        let at = 0x100 + 16 * (i + 1);
        b.u32(at, name);
        b.put(at + 12, &[0x12]);
    }
    b.put(0x180, b"\0f\0g\0");
    // .rela.plt: (r_offset, r_info = sym << 8 | R_X86_64_JUMP_SLOT, r_addend).
    for (i, (slot, sym)) in [(0x3f0u32, 1u32), (0x3f4, 2)].into_iter().enumerate() {
        let at = 0x200 + 12 * i;
        b.u32(at, slot);
        b.u32(at + 4, sym << 8 | 7);
        b.u32(at + 8, 0);
    }
    let dynamic: [(u32, u32); 8] = [
        (6, 0x100),  // DT_SYMTAB
        (5, 0x180),  // DT_STRTAB
        (10, 5),     // DT_STRSZ
        (11, 16),    // DT_SYMENT
        (23, 0x200), // DT_JMPREL
        (2, 24),     // DT_PLTRELSZ
        (20, 7),     // DT_PLTREL = DT_RELA
        (0, 0),
    ];
    for (i, (t, v)) in dynamic.into_iter().enumerate() {
        b.u32(0x300 + 8 * i, t);
        b.u32(0x300 + 8 * i + 4, v);
    }
    b.0
}

#[test]
fn the_plt_relocations_are_rela_or_rel_as_dt_pltrel_says() {
    let bytes = x32();
    let img = Image::load(&bytes).unwrap();
    assert!(!img.is_64());
    let relocs: Vec<_> = img
        .relocs
        .iter()
        .map(|r| (r.offset, r.kind, r.sym, r.addend))
        .collect();
    assert_eq!(
        relocs,
        [
            (0x3f0, RelocKind::JumpSlot, 1, 0),
            (0x3f4, RelocKind::JumpSlot, 2, 0)
        ]
    );
    let names: Vec<_> = img.imports.iter().map(|i| i.name).collect();
    assert_eq!(names, ["f", "g"]);
}

#[test]
fn one_relocation_naming_a_missing_symbol_drops_only_itself() {
    use wazabin_binary::BinaryFormat;
    use wazabin_binary::elf::ElfBinary;

    let mut b = Buf(build(true));
    // The first GLOB_DAT of `.rela.dyn` (`__cxa_finalize`) names symbol 100
    // of an 8-entry `.dynsym`.
    b.u64(0x438 + 8, (100 << 32) | 6);
    let elf = ElfBinary::parse(&b.0).unwrap();
    assert_eq!(elf.import_symbol_name(FINALIZE_SLOT), None);
    assert_eq!(elf.import_symbol_name(TYPEINFO_SLOT), Some("_ZTIi"));
    assert_eq!(elf.import_symbol_name(PUTS_SLOT), Some("puts"));
    assert_eq!(elf.import_symbol_name(FREE_SLOT), Some("free"));
}
