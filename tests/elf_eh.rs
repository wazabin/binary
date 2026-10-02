//! The exception tables on the byte-assembled PIE: the `.eh_frame_hdr` search
//! table, a `zPLR` CIE with an indirect personality, an LSDA with a catch
//! clause through an indirect type-table entry, a cleanup and a call without a
//! landing pad; plus the function seeds, kept apart from the symbols.

mod common;

use common::*;
use wazabin_binary::BinaryFormat;
use wazabin_binary::eh::{Clause, Ptr};
use wazabin_binary::eh_frame::{fdes_from_hdr, stream_starts};
use wazabin_binary::elf::ElfBinary;
use wazabin_binary::image::{Image, SeedSource};

#[test]
fn lsda_call_sites_pads_and_clauses() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let eh = img.eh();
    assert!(eh.warnings.is_empty(), "{:?}", eh.warnings);
    assert_eq!(eh.fdes, 3);
    assert_eq!(eh.funcs.len(), 1, "only the FDE with an LSDA");
    let f = &eh.funcs[0];
    assert_eq!((f.start, f.end, f.lsda), (0x700, 0x710, 0x800));
    // The personality is an indirect pointer through a GLOB_DAT slot: by name.
    assert_eq!(f.personality, Some(Ptr::Sym("__gxx_personality_v0".into())));
    assert_eq!(f.sites.len(), 3);
    let s = &f.sites[0];
    assert_eq!((s.start, s.end, s.pad), (0x700, 0x704, Some(0x710)));
    match &s.clauses[..] {
        [
            Clause::Catch {
                index: 1,
                ty: Ptr::Sym(n),
            },
        ] => assert_eq!(n, "_ZTIi"),
        other => panic!("{other:?}"),
    }
    // A call with no landing pad has no clauses.
    assert_eq!(
        (f.sites[1].start, f.sites[1].pad, f.sites[1].clauses.len()),
        (0x704, None, 0)
    );
    // A pad with no action is a cleanup.
    assert_eq!((f.sites[2].pad, f.sites[2].clauses.len()), (Some(0x720), 1));
    assert!(matches!(f.sites[2].clauses[0], Clause::Cleanup));
    // Read once, kept.
    assert!(std::ptr::eq(img.eh(), eh));
}

#[test]
fn an_emptied_or_missing_header_means_no_exception_handling() {
    let mut bytes = build(false);
    // Zero PT_GNU_EH_FRAME's memsz (what `objcopy --remove-section` does).
    let ph = 64 + 56 * 4;
    bytes[ph + 40..ph + 48].fill(0);
    let img = Image::load(&bytes).unwrap();
    assert!(img.eh_frame_hdr().is_none());
    assert!(img.eh().funcs.is_empty() && img.eh().warnings.is_empty());
}

#[test]
fn a_corrupt_header_warns_instead_of_failing() {
    let mut bytes = build(false);
    bytes[0x900] = 9; // unknown .eh_frame_hdr version
    let img = Image::load(&bytes).unwrap();
    assert!(img.eh().funcs.is_empty());
    assert_eq!(img.eh().warnings.len(), 1);
}

#[test]
fn the_walker_reads_the_stream_with_or_without_the_search_table() {
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let mem = |va: u64| img.bytes_from(va);
    let (eh_frame, fdes) = fdes_from_hdr(&mem, 0x900, 0x24, 8).unwrap();
    assert_eq!(eh_frame, 0x940);
    let mut starts: Vec<_> = fdes.iter().map(|f| f.start).collect();
    starts.sort();
    assert_eq!(starts, [0x600, 0x700, 0x710]);
    assert_eq!(
        fdes.iter().filter(|f| f.lsda.is_some()).count(),
        3,
        "a null LSDA pointer is still a field"
    );
    let mut walked = stream_starts(&mem, 0x940, 8);
    walked.sort();
    assert_eq!(walked, starts);

    // Without a search table (count encoding omitted) the stream is walked.
    let mut no_table = bytes.clone();
    no_table[0x902] = 0xff;
    let img2 = Image::load(&no_table).unwrap();
    let mem2 = |va: u64| img2.bytes_from(va);
    let (_, fdes) = fdes_from_hdr(&mem2, 0x900, 0x24, 8).unwrap();
    assert_eq!(fdes.len(), 3);
}

#[test]
fn seeds_are_metadata_function_starts_without_the_plt_table() {
    let bytes = build(true);
    let img = Image::load(&bytes).unwrap();
    let mut eh_seeds: Vec<_> = img
        .seeds()
        .into_iter()
        .filter(|s| s.source == SeedSource::EhFrame)
        .map(|s| s.addr)
        .collect();
    eh_seeds.sort();
    assert_eq!(
        eh_seeds,
        [0x700, 0x710],
        "the FDE covering the .plt is not a function"
    );
    let mut init: Vec<_> = img
        .seeds()
        .into_iter()
        .filter(|s| s.source == SeedSource::InitFini)
        .map(|s| s.addr)
        .collect();
    init.sort();
    assert_eq!(init, [0x710, 0x718, 0x730, 0x740]);
    // Without section names no PLT range is known.
    let seg = Image::load_segments(&bytes).unwrap();
    assert!(seg.seeds().iter().any(|s| s.addr == 0x600));
    assert_eq!(seg.eh_function_starts().len(), 3);
}

#[test]
fn elf_binary_keeps_seeds_out_of_its_known_functions() {
    let bytes = build(true);
    let elf = ElfBinary::parse(&bytes).unwrap();
    let a = &elf.analysis;
    // Symbols and PLT stubs only: `main`, the stubs the relocations name.
    assert!(
        a.known_functions
            .iter()
            .all(|f| f.name.is_some() || f.is_external)
    );
    assert!(
        a.known_functions
            .iter()
            .any(|f| f.name.as_deref() == Some("main"))
    );
    assert!(
        !a.known_functions
            .iter()
            .any(|f| f.address == 0x730 || f.address == 0x740)
    );
    // The seeds are listed apart, and still seed `entry_points`.
    let seeds: Vec<u64> = a.seeds.iter().map(|s| s.addr).collect();
    assert!(seeds.contains(&0x730) && seeds.contains(&0x740) && seeds.contains(&0x710));
    let entries = elf.entry_points();
    assert!(entries.contains(&0x730) && entries.contains(&0x740));
    // They never get a name.
    assert_eq!(elf.symbol_name(0x730), None);
}
