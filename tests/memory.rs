//! Zero fill is sized by header values: a few hundred bytes of file may ask
//! for terabytes of memory. The allocator here refuses any one allocation
//! over a few hundred MiB, so a reader that trusts such a size aborts the
//! test instead of paging the memory in.

mod common;
#[path = "../src/test_alloc.rs"]
mod test_alloc;

use common::*;
use wazabin_binary::BinaryFormat;
use wazabin_binary::elf::ElfBinary;
use wazabin_binary::image::Image;

#[test]
fn a_terabyte_bss_is_not_materialized() {
    let mut b = Buf(build(true));
    b.u64(64 + 56 + 40, 1 << 40); // the RW segment's p_memsz
    let bytes = b.0;
    let img = Image::load(&bytes).unwrap();
    assert_eq!(img.segments[1].memsz, 1 << 40);

    let elf = ElfBinary::parse(&bytes).unwrap();
    assert_eq!(elf.byte_at(0x3000 + (1 << 39)), Some(0));
    let regions = elf.mapped_regions();
    let file_backed: usize = elf.segments.iter().map(|s| s.data.len()).sum();
    let total: usize = regions.iter().map(|r| r.1.len()).sum();
    assert!(total - file_backed <= wazabin_binary::ZERO_FILL_LIMIT as usize);
    // The file bytes are all there.
    assert_eq!(regions[1].1[..8], bytes[0x1000..0x1008]);
}
