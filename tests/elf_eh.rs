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

/// The unwind tables are located by `PT_GNU_EH_FRAME` and, with section headers, by the names of
/// `.eh_frame` and `.gcc_except_table`.
#[test]
fn unwind_tables_with_and_without_section_headers() {
    let bytes = build(true);
    let img = Image::load(&bytes).unwrap();
    assert_eq!(
        img.unwind_tables(),
        vec![(0x800, 0x820), (0x900, 0x924), (0x940, 0x9e0)]
    );
    let bytes = build(false);
    let img = Image::load(&bytes).unwrap();
    let hdr: Vec<(u64, u64)> = img
        .eh_frame_hdr()
        .map(|(a, n)| (a, a + n))
        .into_iter()
        .collect();
    assert_eq!(
        img.unwind_tables(),
        hdr,
        "without section headers, only the header the program headers name"
    );
}

/// A hostile LSDA: `n` `.eh_frame_hdr` entries for one FDE whose LSDA has
/// 65536 call sites, every one landing on the same action chain: `chain`
/// cleanups, or with `chain` 0 one record whose `next` points back at
/// itself. Read naively the latter is 256 clauses a site, a site list per
/// header entry: gigabytes from a 262 KB file.
fn hostile_lsda(n: u32, chain: usize) -> Vec<u8> {
    const B: u64 = 0x40_0000;
    let hdr_off = 64 + 2 * 56;
    let hdr_len = 12 + 8 * n as usize;
    let ehf_off = hdr_off + hdr_len;
    // CIE: version 1, "zL", code/data align 1/-8, RA 16, augmentation data:
    // the LSDA encoding (udata4).
    let cie_body = [
        &0u32.to_le_bytes()[..],
        &[1],
        b"zL\0",
        &[1, 0x78, 16, 1, 0x03],
    ]
    .concat();
    let cie = [&(cie_body.len() as u32).to_le_bytes()[..], &cie_body].concat();
    let fde_off = ehf_off + cie.len();
    let lsda_off = fde_off + 4 + 4 + 8 + 8 + 1 + 4;
    let fde_body = [
        &((fde_off + 4 - ehf_off) as u32).to_le_bytes()[..],
        &(B + 0x10).to_le_bytes(),
        &0x10u64.to_le_bytes(),
        &[4],
        &((B as usize + lsda_off) as u32).to_le_bytes(),
    ]
    .concat();
    let fde = [&(fde_body.len() as u32).to_le_bytes()[..], &fde_body].concat();
    const SITES: usize = 65536;
    let cs_len = 4 * SITES;
    let mut lsda = vec![0xff, 0xff, 0x01];
    let mut v = cs_len;
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        lsda.push(b | if v != 0 { 0x80 } else { 0 });
        if v == 0 {
            break;
        }
    }
    // Each site: start 1, len 1, pad 1, action 1 (uleb).
    lsda.extend(std::iter::repeat_n(1u8, cs_len));
    if chain == 0 {
        // The action record: filter 0 (cleanup), next -1 (itself).
        lsda.extend([0x00, 0x7f]);
    } else {
        // Cleanups, each `next` the record after it, the last 0.
        for i in 0..chain {
            lsda.extend([0x00, u8::from(i + 1 < chain)]);
        }
    }
    lsda.extend([0; 16]);
    let mut hdr = vec![1, 0x03, 0x03, 0x03];
    hdr.extend(((B as usize + ehf_off) as u32).to_le_bytes());
    hdr.extend(n.to_le_bytes());
    for _ in 0..n {
        hdr.extend(((B + 0x10) as u32).to_le_bytes());
        hdr.extend(((B as usize + fde_off) as u32).to_le_bytes());
    }
    let body = [hdr, cie, fde].concat();
    let total = (hdr_off + body.len() + lsda.len()) as u64;
    let mut out = b"\x7fELF".to_vec();
    out.extend([2, 1, 1, 0]);
    out.extend([0; 8]);
    out.extend(3u16.to_le_bytes()); // ET_DYN
    out.extend(62u16.to_le_bytes());
    out.extend(1u32.to_le_bytes());
    out.extend((B + 0x10).to_le_bytes());
    out.extend(64u64.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    for h in [64u16, 56, 2, 64, 0, 0] {
        out.extend(h.to_le_bytes());
    }
    let phdr = |kind: u32, flags: u32, off: u64, va: u64, sz: u64, align: u64| {
        let mut p = kind.to_le_bytes().to_vec();
        p.extend(flags.to_le_bytes());
        for f in [off, va, va, sz, sz, align] {
            p.extend(f.to_le_bytes());
        }
        p
    };
    out.extend(phdr(1, 5, 0, B, total, 0x1000));
    out.extend(phdr(
        0x6474_e550,
        4,
        hdr_off as u64,
        B + hdr_off as u64,
        hdr_len as u64,
        4,
    ));
    out.extend(body);
    out.extend(lsda);
    out
}

#[test]
fn a_self_referencing_action_chain_is_read_once() {
    let bytes = hostile_lsda(8, 0);
    let img = Image::load(&bytes).unwrap();
    let eh = img.eh();
    assert_eq!(eh.fdes, 8);
    assert_eq!(eh.funcs.len(), 1);
    let f = &eh.funcs[0];
    assert_eq!(f.sites.len(), 65536);
    // The chain stops where it comes back: one cleanup, not 256.
    assert!(
        f.sites.iter().all(|s| s.clauses.len() == 1),
        "{:?}",
        f.sites[0].clauses.len()
    );
}

#[test]
fn more_clauses_than_the_budget_are_a_warning() {
    // 65536 sites sharing a chain of 255 cleanups: 16M clauses, in 260 KB.
    let bytes = hostile_lsda(1, 255);
    let img = Image::load(&bytes).unwrap();
    let eh = img.eh();
    assert!(eh.funcs.is_empty());
    assert_eq!(eh.warnings.len(), 1, "{:?}", eh.warnings);
    assert!(
        eh.warnings[0].contains("the rest not lifted"),
        "{:?}",
        eh.warnings
    );
    // Under the budget, the same chain is read whole.
    let bytes = hostile_lsda(1, 32);
    let img = Image::load(&bytes).unwrap();
    let eh = img.eh();
    assert!(eh.warnings.is_empty(), "{:?}", eh.warnings);
    assert!(eh.funcs[0].sites.iter().all(|s| s.clauses.len() == 32));
}
