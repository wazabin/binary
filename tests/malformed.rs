//! Malformed input: every header value is untrusted, so no field, however
//! large, may make a reader panic (the test profile checks for overflow) or
//! wrap into a different answer. Each test feeds the readers a broken file and
//! asks only that they return.

mod common;

use common::*;
use wazabin_binary::BinaryFormat;
use wazabin_binary::elf::ElfBinary;
use wazabin_binary::image::Image;

/// Everything the ELF readers compute from a file: the load, the queries
/// that walk tables on demand, and the owned `ElfBinary` view.
fn exercise(bytes: &[u8]) {
    for img in [Image::load(bytes), Image::load_segments(bytes)]
        .into_iter()
        .flatten()
    {
        let _ = img.headers();
        let _ = img.metadata();
        let _ = img.unwind_tables();
        let _ = img.seeds();
        let _ = img.eh();
        for r in &img.relocs {
            let _ = img.word(r.offset);
        }
    }
    if let Ok(elf) = ElfBinary::parse(bytes) {
        let _ = elf.entry_points();
        let _ = elf.sections();
        let _ = elf.symbols();
        let _ = elf.segment_bounds(elf.load_address);
    }
    let _ = wazabin_binary::load(bytes);
}

/// A 0x100-byte ELF64 whose one `PT_LOAD` sits 16 bytes below the top of the
/// address space and holds a `jmp *disp(%rip)`.
fn load_at_the_top_of_the_address_space() -> Vec<u8> {
    let mut b = Buf(vec![0u8; 0x100]);
    b.put(0, b"\x7fELF");
    b.put(4, &[2, 1, 1]);
    b.u16(16, 2); // ET_EXEC
    b.u16(18, 62);
    b.u32(20, 1);
    b.u64(24, 0xffff_ffff_ffff_fff0);
    b.u64(32, 64);
    b.u16(52, 64);
    b.u16(54, 56);
    b.u16(56, 1);
    b.u32(64, 1); // PT_LOAD
    b.u32(68, 5); // R+X
    b.u64(72, 0);
    b.u64(80, 0xffff_ffff_ffff_fff0);
    b.u64(88, 0xffff_ffff_ffff_fff0);
    b.u64(96, 0x100);
    b.u64(104, 0x100);
    b.put(0xf0, &[0xff, 0x25]);
    b.0
}

#[test]
fn a_segment_past_the_top_of_the_address_space_is_an_error() {
    let bytes = load_at_the_top_of_the_address_space();
    assert!(Image::load(&bytes).is_err());
    assert!(ElfBinary::parse(&bytes).is_err());
    assert!(wazabin_binary::load(&bytes).is_err());
}

/// The test image with one program header field replaced.
fn with_phdr_field(index: usize, field: usize, value: u64) -> Vec<u8> {
    let mut b = Buf(build(true));
    b.u64(64 + 56 * index + field, value);
    b.0
}

#[test]
fn huge_header_values_are_not_arithmetic() {
    // `e_phoff` near the top of `usize`.
    let mut b = Buf(build(true));
    b.u64(32, u64::MAX - 8);
    exercise(&b.0);
    // `e_shoff` likewise.
    let mut b = Buf(build(true));
    b.u64(40, u64::MAX - 8);
    exercise(&b.0);
    // A note, a `PT_INTERP` and a `PT_PHDR` that end past the address space.
    for kind in [3u64, 4, 6] {
        let mut b = Buf(with_phdr_field(5, 16, u64::MAX - 4));
        b.u32(64 + 56 * 5, kind as u32);
        exercise(&b.0);
    }
}

/// The dynamic table of the test image with the value of `tag` replaced.
fn with_dynamic(tag: i64, value: u64) -> Vec<u8> {
    let mut b = build(false);
    for i in 0..21 {
        let at = 0x1000 + 16 * i;
        if i64::from_le_bytes(b[at..at + 8].try_into().unwrap()) == tag {
            b[at + 8..at + 16].copy_from_slice(&value.to_le_bytes());
        }
    }
    b
}

#[test]
fn huge_dynamic_values_are_not_arithmetic() {
    for tag in [2, 5, 6, 7, 8, 10, 11, 23, 25, 27, 0x6fff_fff0, 0x6fff_fffe] {
        for value in [u64::MAX, u64::MAX - 0x10, 1 << 63, 1] {
            exercise(&with_dynamic(tag, value));
        }
    }
    // A string table far above a symbol table counts more symbols than a
    // `usize` of bytes holds.
    let mut b = with_dynamic(5, u64::MAX);
    b[0x1000 + 16 * 8 + 8..0x1000 + 16 * 8 + 16].copy_from_slice(&1u64.to_le_bytes());
    exercise(&b);
}

/// Replace the LSDA's action record with `filter`, `next` (both sleb128).
fn with_action(filter: &[u8], next: &[u8]) -> Vec<u8> {
    let mut b = build(false);
    let actions = 0x800 + 5 + 12;
    b[actions..actions + filter.len()].copy_from_slice(filter);
    b[actions + filter.len()..actions + filter.len() + next.len()].copy_from_slice(next);
    b
}

#[test]
fn extreme_lsda_indices_are_unreadable_not_arithmetic() {
    let min = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f];
    let max = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
    for (filter, next) in [
        (&min[..], &[0][..]),    // a filter of i64::MIN
        (&max[..], &[0][..]),    // a type index of i64::MAX
        (&[0x01][..], &max[..]), // a next offset of i64::MAX
    ] {
        let bytes = with_action(filter, next);
        let img = Image::load(&bytes).unwrap();
        let _ = img.eh();
    }
    // A call site whose action index is u64::MAX, and a type-table offset
    // that runs off the address space.
    let mut b = build(false);
    b[0x800 + 5 + 3] = 0x7f; // action 0x7f, far past the table
    exercise(&b);
    let mut b = build(false);
    b[0x802..0x80c].copy_from_slice(&max);
    exercise(&b);
}

/// A tiny deterministic generator, so a failure names its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

#[test]
fn truncated_and_mutated_images_never_panic() {
    for sections in [false, true] {
        let base = build(sections);
        // Every truncation on a 16-byte grid.
        for len in (0..base.len()).step_by(16) {
            exercise(&base[..len]);
        }
        // Every aligned field set to the values a corrupted one most often
        // takes, as a 32-bit and as a 64-bit field. With a section table,
        // only the bytes it adds.
        let from = if sections { 0x1408 } else { 0 };
        for at in (from..base.len()).step_by(4) {
            for v in [0, 0x8000_0000, u32::MAX] {
                let mut b = base.clone();
                let n = 4.min(b.len() - at);
                b[at..at + n].copy_from_slice(&v.to_le_bytes()[..n]);
                exercise(&b);
            }
            if at % 8 == 0 && at + 8 <= base.len() {
                for v in [1 << 63, u64::MAX - 0xf, u64::MAX] {
                    let mut b = base.clone();
                    b[at..at + 8].copy_from_slice(&u64::to_le_bytes(v));
                    exercise(&b);
                }
            }
        }
        // Random damage, a few bytes at a time.
        let mut rng = Rng(0x5eed_0000 + sections as u64);
        for _ in 0..500 {
            let mut b = base.clone();
            for _ in 0..1 + rng.next() % 8 {
                let at = (rng.next() as usize) % b.len();
                b[at] = rng.next() as u8;
            }
            exercise(&b);
        }
    }
}
