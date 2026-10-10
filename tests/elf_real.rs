//! The loader view on a real binary: the test executable itself. No fixture is
//! checked in; whatever linker built this test is the one under test, so the
//! assertions are the invariants any linked ELF has, not byte values.

#![cfg(target_os = "linux")]

use wazabin_binary::BinaryFormat;
use wazabin_binary::elf::ElfBinary;
use wazabin_binary::image::{Image, RelocKind, SymbolKind};

fn exe() -> Vec<u8> {
    std::fs::read(std::env::current_exe().unwrap()).unwrap()
}

#[test]
fn segments_entry_and_stubs_are_consistent() {
    let bytes = exe();
    let img = Image::load(&bytes).unwrap();
    assert!(!img.segments.is_empty());
    assert!(
        img.segments
            .iter()
            .any(|s| s.executable() && s.contains(img.entry)),
        "the entry point is in code"
    );
    // Every PLT stub the scan found is in executable memory and jumps through
    // a slot some relocation fills.
    for i in &img.imports {
        if let (Some(stub), Some(slot)) = (i.stub, i.slot) {
            assert!(
                img.segments
                    .iter()
                    .any(|s| s.executable() && s.contains(stub)),
                "{} stub {stub:#x}",
                i.name
            );
            assert!(img.reloc_at(slot).is_some(), "{} slot {slot:#x}", i.name);
            assert_eq!(img.import_name(stub), Some(i.name));
        }
    }
    // A section-less read agrees on everything the dynamic table carries.
    let seg = Image::load_segments(&bytes).unwrap();
    assert!(seg.sections.is_empty());
    assert_eq!(seg.segments, img.segments);
    assert_eq!(seg.dynamic.needed, img.dynamic.needed);
    // ... and names every relocation's symbol (the count of `.dynsym` comes from the hash tables and the
    // relocations then, not the section).
    for r in &img.relocs {
        assert_eq!(seg.reloc_symbol(r).map(|s| s.name), img.reloc_symbol(r).map(|s| s.name), "relocation at {:#x}", r.offset);
    }
    assert_eq!(seg.dynsyms(), img.dynsyms());
    assert_eq!(seg.relocs.len(), img.relocs.len());
    assert_eq!(seg.imports.len(), img.imports.len());
}

#[test]
fn relative_relocations_point_into_the_image() {
    let bytes = exe();
    let img = Image::load(&bytes).unwrap();
    if !img.is_pie() {
        return;
    }
    let relative: Vec<_> = img
        .relocs
        .iter()
        .filter(|r| r.kind == RelocKind::Relative)
        .collect();
    assert!(!relative.is_empty(), "a PIE has RELATIVE relocations");
    let in_image = |a: u64| img.segments.iter().any(|s| s.contains(a));
    assert!(
        relative.iter().all(|r| in_image(r.offset)),
        "every relocated word is in a segment"
    );
    // `word` applies them: the addend, not the (zero) file word.
    let r = relative[0];
    assert_eq!(img.word(r.offset), Some(r.addend as u64));
}

#[test]
fn dynsym_agrees_with_elf_binary() {
    let bytes = exe();
    let img = Image::load(&bytes).unwrap();
    let elf = ElfBinary::parse(&bytes).unwrap();
    for s in img
        .dynsyms()
        .iter()
        .filter(|s| s.kind == SymbolKind::Func && s.defined && s.addr != 0 && !s.name.is_empty())
    {
        assert!(elf.symbol_name(s.addr).is_some(), "{}", s.name);
    }
    assert_eq!(elf.entrypoint(), Some(img.entry));
    assert_eq!(elf.linked_libraries(), img.dynamic.needed);
}

#[test]
fn exception_tables_read_on_demand() {
    let bytes = exe();
    let img = Image::load(&bytes).unwrap();
    let starts = img.eh_function_starts();
    assert!(!starts.is_empty(), "a Rust binary has unwind tables");
    let eh = img.eh();
    assert!(eh.warnings.is_empty(), "{:?}", eh.warnings);
    assert!(eh.fdes >= eh.funcs.len());
    assert!(!eh.funcs.is_empty(), "and landing pads");
    for f in &eh.funcs {
        assert!(f.start < f.end);
        for s in &f.sites {
            assert!(
                s.start >= f.start && s.start < s.end,
                "a call site starts in its function"
            );
        }
    }
    // Seeds are executable.
    assert!(img.seeds().iter().all(|s| {
        img.segments
            .iter()
            .any(|g| g.executable() && g.contains(s.addr))
    }));
}
