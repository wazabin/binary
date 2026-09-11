//! Function starts from the ELF `.eh_frame` unwind tables.
//!
//! Every function compiled with unwind tables (the default for x86-64 gcc and
//! clang, `-fasynchronous-unwind-tables`) gets one FDE whose `initial_location`
//! is the function's first byte. Both `.eh_frame` and `.eh_frame_hdr` are
//! allocated sections, so `strip` keeps them: they are the best function-start
//! source a stripped binary has.
//!
//! Two readers, tried in order:
//! - `.eh_frame_hdr`'s binary-search table, which lists every FDE's initial
//!   location directly (the linker builds it with `--eh-frame-hdr`).
//! - a walk over `.eh_frame`'s CIE/FDE chain, decoding each FDE's
//!   `initial_location` with its CIE's pointer encoding, for binaries without
//!   the header table.
//!
//! Only starts are reported: the analysis grows each function itself.

use elf::{ElfBytes, endian::AnyEndian};

/// A byte cursor over one section, with the DWARF pointer-encoding readers.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Virtual address of `bytes[0]`, for `DW_EH_PE_pcrel`.
    base: u64,
    /// Pointer width in bytes, for `DW_EH_PE_absptr`.
    ptr_size: usize,
}

const DW_EH_PE_ABSPTR: u8 = 0x00;
const DW_EH_PE_ULEB128: u8 = 0x01;
const DW_EH_PE_UDATA2: u8 = 0x02;
const DW_EH_PE_UDATA4: u8 = 0x03;
const DW_EH_PE_UDATA8: u8 = 0x04;
const DW_EH_PE_SLEB128: u8 = 0x09;
const DW_EH_PE_SDATA2: u8 = 0x0a;
const DW_EH_PE_SDATA4: u8 = 0x0b;
const DW_EH_PE_SDATA8: u8 = 0x0c;
const DW_EH_PE_PCREL: u8 = 0x10;
const DW_EH_PE_DATAREL: u8 = 0x30;
const DW_EH_PE_OMIT: u8 = 0xff;

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], base: u64, ptr_size: usize) -> Self {
        Self { bytes, pos: 0, base, ptr_size }
    }

    fn address(&self) -> u64 {
        self.base.wrapping_add(self.pos as u64)
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn uleb128(&mut self) -> Option<u64> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift < 64 {
                result |= u64::from(byte & 0x7f) << shift;
            }
            shift += 7;
            if byte & 0x80 == 0 {
                return Some(result);
            }
        }
    }

    fn sleb128(&mut self) -> Option<i64> {
        let mut result = 0i64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift < 64 {
                result |= i64::from(byte & 0x7f) << shift;
            }
            shift += 7;
            if byte & 0x80 == 0 {
                if shift < 64 && byte & 0x40 != 0 {
                    result |= -1i64 << shift;
                }
                return Some(result);
            }
        }
    }

    /// Read one pointer with `encoding`; `datarel` is the base for
    /// `DW_EH_PE_datarel` (the `.eh_frame_hdr` address when reading the header).
    fn encoded(&mut self, encoding: u8, datarel: u64) -> Option<u64> {
        if encoding == DW_EH_PE_OMIT {
            return None;
        }
        let at = self.address();
        let raw: u64 = match encoding & 0x0f {
            DW_EH_PE_ABSPTR => match self.ptr_size {
                4 => u64::from(self.u32()?),
                _ => self.u64()?,
            },
            DW_EH_PE_ULEB128 => self.uleb128()?,
            DW_EH_PE_UDATA2 => u64::from(self.u16()?),
            DW_EH_PE_UDATA4 => u64::from(self.u32()?),
            DW_EH_PE_UDATA8 => self.u64()?,
            DW_EH_PE_SLEB128 => self.sleb128()? as u64,
            DW_EH_PE_SDATA2 => i64::from(self.u16()? as i16) as u64,
            DW_EH_PE_SDATA4 => i64::from(self.u32()? as i32) as u64,
            DW_EH_PE_SDATA8 => self.u64()?,
            _ => return None,
        };
        let value = match encoding & 0x70 {
            0x00 => raw,
            DW_EH_PE_PCREL => at.wrapping_add(raw),
            DW_EH_PE_DATAREL => datarel.wrapping_add(raw),
            // textrel / funcrel / aligned: not produced by the toolchains we
            // care about, and we have no base for them.
            _ => return None,
        };
        // The indirect bit (0x80) means "the pointer lives at that address";
        // only the personality routine uses it, which we never read.
        Some(value)
    }
}

/// Locate a section by name: `(virtual address, file bytes)`.
fn section<'a>(
    elf: &ElfBytes<'a, AnyEndian>,
    file: &'a [u8],
    name: &str,
) -> Option<(u64, &'a [u8])> {
    let (shdrs, shstrtab) = match elf.section_headers_with_strtab() {
        Ok((Some(s), Some(st))) => (s, st),
        _ => return None,
    };
    let shdr = shdrs
        .iter()
        .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(name))?;
    if shdr.sh_addr == 0 || shdr.sh_type == elf::abi::SHT_NOBITS {
        return None;
    }
    let start = shdr.sh_offset as usize;
    let end = start.checked_add(shdr.sh_size as usize)?;
    Some((shdr.sh_addr, file.get(start..end)?))
}

/// Function starts from `.eh_frame_hdr` / `.eh_frame`, unsorted, possibly
/// with duplicates. Empty when the binary carries neither section or the
/// tables are malformed.
pub fn function_starts(elf: &ElfBytes<'_, AnyEndian>, file: &[u8]) -> Vec<u64> {
    let ptr_size = match elf.ehdr.class {
        elf::file::Class::ELF32 => 4,
        elf::file::Class::ELF64 => 8,
    };
    if let Some((addr, bytes)) = section(elf, file, ".eh_frame_hdr")
        && let Some(starts) = from_hdr(bytes, addr, ptr_size)
    {
        return starts;
    }
    match section(elf, file, ".eh_frame") {
        Some((addr, bytes)) => from_eh_frame(bytes, addr, ptr_size),
        None => Vec::new(),
    }
}

/// The `.eh_frame_hdr` search table: `(initial_location, fde_address)` pairs.
fn from_hdr(bytes: &[u8], addr: u64, ptr_size: usize) -> Option<Vec<u64>> {
    let mut r = Reader::new(bytes, addr, ptr_size);
    if r.u8()? != 1 {
        return None;
    }
    let eh_frame_ptr_enc = r.u8()?;
    let fde_count_enc = r.u8()?;
    let table_enc = r.u8()?;
    let _eh_frame_ptr = r.encoded(eh_frame_ptr_enc, addr);
    if fde_count_enc == DW_EH_PE_OMIT || table_enc == DW_EH_PE_OMIT {
        return None;
    }
    let count = r.encoded(fde_count_enc, addr)?;
    let mut starts = Vec::with_capacity(count.min(1 << 20) as usize);
    for _ in 0..count {
        let start = r.encoded(table_enc, addr)?;
        let _fde = r.encoded(table_enc, addr)?;
        starts.push(start);
    }
    Some(starts)
}

/// Per-CIE facts an FDE needs to decode its `initial_location`.
#[derive(Clone, Copy)]
struct Cie {
    /// The `R` augmentation's pointer encoding (`DW_EH_PE_absptr` without one).
    fde_encoding: u8,
}

/// Parse the CIE whose header starts at `pos` (just past its length field and
/// the zero CIE id).
fn parse_cie(r: &mut Reader<'_>) -> Option<Cie> {
    let version = r.u8()?;
    if version != 1 && version != 3 && version != 4 {
        return None;
    }
    let mut augmentation = Vec::new();
    loop {
        let byte = r.u8()?;
        if byte == 0 {
            break;
        }
        augmentation.push(byte);
    }
    if augmentation.first() == Some(&b'e') && augmentation.get(1) == Some(&b'h') {
        // Old gcc "eh" augmentation carries a pointer before the alignment
        // factors.
        r.take(r.ptr_size)?;
    }
    if version >= 4 {
        let _address_size = r.u8()?;
        let _segment_size = r.u8()?;
    }
    let _code_align = r.uleb128()?;
    let _data_align = r.sleb128()?;
    let _return_reg = if version == 1 { u64::from(r.u8()?) } else { r.uleb128()? };
    let mut cie = Cie { fde_encoding: DW_EH_PE_ABSPTR };
    if augmentation.first() == Some(&b'z') {
        let _augmentation_len = r.uleb128()?;
        for &byte in &augmentation[1..] {
            match byte {
                b'R' => cie.fde_encoding = r.u8()?,
                b'P' => {
                    let enc = r.u8()?;
                    r.encoded(enc & 0x7f, 0)?;
                }
                b'L' => {
                    r.u8()?;
                }
                b'S' | b'B' | b'G' => {}
                _ => return None,
            }
        }
    }
    Some(cie)
}

/// Walk `.eh_frame`, returning each FDE's `initial_location`.
fn from_eh_frame(bytes: &[u8], addr: u64, ptr_size: usize) -> Vec<u64> {
    let mut starts = Vec::new();
    let mut cies: Vec<(usize, Cie)> = Vec::new();
    let mut r = Reader::new(bytes, addr, ptr_size);
    loop {
        let entry_start = r.pos;
        let Some(mut length) = r.u32().map(u64::from) else { break };
        if length == 0 {
            // Zero terminator.
            break;
        }
        if length == 0xffff_ffff {
            let Some(l) = r.u64() else { break };
            length = l;
        }
        let body_start = r.pos;
        let Some(body_end) = body_start.checked_add(length as usize) else { break };
        if body_end > bytes.len() {
            break;
        }
        let Some(id) = r.u32() else { break };
        if id == 0 {
            if let Some(cie) = parse_cie(&mut r) {
                cies.push((entry_start, cie));
            }
        } else {
            // `id` is the distance back from its own field to the owning CIE.
            let id_pos = body_start;
            let cie_pos = id_pos.wrapping_sub(id as usize);
            if let Some((_, cie)) = cies.iter().find(|(pos, _)| *pos == cie_pos)
                && let Some(start) = r.encoded(cie.fde_encoding, 0)
            {
                // A zero-length FDE (a discarded section's remnant) is not a
                // function; its range follows with the same encoding's low
                // nibble and no relocation.
                let range = r.encoded(cie.fde_encoding & 0x0f, 0).unwrap_or(0);
                if range != 0 {
                    starts.push(start);
                }
            }
        }
        r.pos = body_end;
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode one `.eh_frame` with a `zR` CIE (pcrel|sdata4) and two FDEs, and
    /// check the chain walk finds their initial locations.
    fn synthetic_eh_frame() -> (Vec<u8>, u64) {
        let base = 0x40_0000u64;
        let mut out = Vec::new();
        // CIE: length, id=0, version 1, "zR", code align 1, data align -8,
        // return reg 16, aug len 1, R = pcrel|sdata4, then padding to 8.
        let mut cie = vec![0u8; 4]; // length placeholder
        cie.extend_from_slice(&0u32.to_le_bytes());
        cie.push(1);
        cie.extend_from_slice(b"zR\0");
        cie.push(1);
        cie.push(0x78); // sleb128 -8
        cie.push(16);
        cie.push(1);
        cie.push(DW_EH_PE_PCREL | DW_EH_PE_SDATA4);
        while (cie.len()) % 8 != 0 {
            cie.push(0);
        }
        let len = (cie.len() - 4) as u32;
        cie[..4].copy_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&cie);

        for target in [0x40_1000u64, 0x40_2000u64] {
            let fde_start = out.len();
            let mut fde = vec![0u8; 4];
            let id_pos = fde_start + 4;
            fde.extend_from_slice(&((id_pos - 0) as u32).to_le_bytes()); // CIE at offset 0
            let field_addr = base + (fde_start + 8) as u64;
            let rel = (target as i64 - field_addr as i64) as i32;
            fde.extend_from_slice(&rel.to_le_bytes());
            fde.extend_from_slice(&0x20u32.to_le_bytes()); // range
            fde.push(0); // aug len
            while fde.len() % 8 != 0 {
                fde.push(0);
            }
            let len = (fde.len() - 4) as u32;
            fde[..4].copy_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&fde);
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        (out, base)
    }

    #[test]
    fn walks_cie_fde_chain() {
        let (bytes, base) = synthetic_eh_frame();
        assert_eq!(from_eh_frame(&bytes, base, 8), vec![0x40_1000, 0x40_2000]);
    }

    #[test]
    fn reads_hdr_table() {
        // version 1, eh_frame_ptr pcrel|sdata4, count udata4, table datarel|sdata4.
        let base = 0x40_8000u64;
        let mut hdr = vec![1, DW_EH_PE_PCREL | DW_EH_PE_SDATA4, DW_EH_PE_UDATA4, DW_EH_PE_DATAREL | DW_EH_PE_SDATA4];
        hdr.extend_from_slice(&(-0x1000i32).to_le_bytes());
        hdr.extend_from_slice(&2u32.to_le_bytes());
        for (start, fde) in [(0x40_1000i64, 0x40_7010i64), (0x40_2000, 0x40_7030)] {
            hdr.extend_from_slice(&((start - base as i64) as i32).to_le_bytes());
            hdr.extend_from_slice(&((fde - base as i64) as i32).to_le_bytes());
        }
        assert_eq!(from_hdr(&hdr, base, 8), Some(vec![0x40_1000, 0x40_2000]));
    }

    #[test]
    fn leb128_round_trips() {
        let mut r = Reader::new(&[0xe5, 0x8e, 0x26, 0x7f, 0x80, 0x7f], 0, 8);
        assert_eq!(r.uleb128(), Some(624485));
        assert_eq!(r.sleb128(), Some(-1));
        assert_eq!(r.sleb128(), Some(-128));
    }
}
