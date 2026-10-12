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
