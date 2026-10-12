//! The `.eh_frame` reader: one byte cursor, one CIE / FDE / `.eh_frame_hdr`
//! walker.
//!
//! Every function compiled with unwind tables (the default for x86-64 gcc and
//! clang) has one FDE whose `initial_location` is the function's first byte.
//! `.eh_frame` and `.eh_frame_hdr` are allocated, so `strip` keeps them.
//!
//! The walker reads memory through a [`Mem`] function (the bytes from an
//! address to the end of whatever holds it), so it serves a loaded
//! [`Image`](crate::image::Image) as well as a bare section. Two ways in:
//!
//! - [`fdes_from_hdr`]: the `.eh_frame_hdr` search table lists every FDE (the
//!   linker builds it with `--eh-frame-hdr`); without a table the stream it
//!   names is walked instead;
//! - [`walk_stream`]: the CIE / FDE chain of `.eh_frame` itself, for binaries
//!   without the header.
//!
//! [`hdr_starts`] and [`stream_starts`] read only the function starts, which
//! is all a seed list needs.
//!
//! Nothing here runs at load time: the LSDA reader of [`crate::eh`] and the
//! seeds of [`Image::seeds`](crate::image::Image::seeds) call it on demand.

use std::collections::BTreeMap;

/// `DW_EH_PE_omit`: the field is absent.
pub const OMIT: u8 = 0xff;

/// A byte cursor with the DWARF pointer-encoding readers.
pub struct Cursor<'a> {
    pub b: &'a [u8],
    pub p: usize,
    /// Address of `b[0]`, for `DW_EH_PE_pcrel`.
    pub base: u64,
    /// Base of `DW_EH_PE_datarel` pointers (the `.eh_frame_hdr` start).
    pub datarel: u64,
    /// Pointer width in bytes, for `DW_EH_PE_absptr`.
    pub ptr_size: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(b: &'a [u8], base: u64) -> Self {
        Cursor {
            b,
            p: 0,
            base,
            datarel: 0,
            ptr_size: 8,
        }
    }

    pub fn addr(&self) -> u64 {
        self.base.wrapping_add(self.p as u64)
    }

    pub fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }

    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }

    pub fn u16(&mut self) -> Option<u64> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?) as u64)
    }

    pub fn u32(&mut self) -> Option<u64> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?) as u64)
    }

    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    pub fn uleb(&mut self) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= u64::from(b & 0x7f) << shift;
            }
            if b & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
        }
    }

    pub fn sleb(&mut self) -> Option<i64> {
        let (mut v, mut shift) = (0i64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= i64::from(b & 0x7f) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    v |= -1i64 << shift;
                }
                return Some(v);
            }
        }
    }

    /// A NUL-terminated string.
    pub fn cstr(&mut self) -> Option<String> {
        let start = self.p;
        while *self.b.get(self.p)? != 0 {
            self.p += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.p]).into_owned();
        self.p += 1;
        Some(s)
    }

    /// A `DW_EH_PE_*`-encoded pointer, with its application (pc-relative to
    /// its own position, or relative to `datarel`) applied. The indirect flag
    /// is left to the caller: the result is then the address of the slot
    /// holding the pointer.
    pub fn encoded(&mut self, enc: u8) -> Option<u64> {
        if enc == OMIT {
            return None;
        }
        let at = self.addr();
        let v = match enc & 0x0f {
            0x00 => match self.ptr_size {
                4 => self.u32()?,
                _ => self.u64()?,
            },
            0x04 | 0x0c => self.u64()?,
            0x01 => self.uleb()?,
            0x02 => self.u16()?,
            0x03 => self.u32()?,
            0x09 => self.sleb()? as u64,
            0x0a => self.u16()? as u16 as i16 as i64 as u64,
            0x0b => self.u32()? as u32 as i32 as i64 as u64,
            _ => return None,
        };
        // A raw 0 is a null pointer whatever the application (libgcc's rule).
        if v == 0 {
            return Some(0);
        }
        match enc & 0x70 {
            0x00 => Some(v),
            0x10 => Some(at.wrapping_add(v)),
            0x30 => Some(self.datarel.wrapping_add(v)),
            _ => None, // textrel / funcrel: not produced for ELF
        }
    }
}

/// The size in bytes of a fixed-width encoding, 0 for a variable-length one.
pub fn enc_size(enc: u8, ptr_size: usize) -> usize {
    match enc & 0x0f {
        0x00 => ptr_size,
        0x04 | 0x0c => 8,
        0x02 | 0x0a => 2,
        0x03 | 0x0b => 4,
        _ => 0,
    }
}

/// A CIE, as the FDEs need it.
#[derive(Clone, Debug)]
pub struct RawCie {
    pub fde_enc: u8,
    pub lsda_enc: u8,
    /// `(encoding, value)` of the `P` augmentation: the value is a slot
    /// address when the encoding is indirect.
    pub personality: Option<(u8, u64)>,
}

/// One FDE, unresolved.
#[derive(Clone, Debug)]
pub struct RawFde {
    /// Address of the record.
    pub addr: u64,
    pub start: u64,
    pub end: u64,
    /// The LSDA pointer as encoded: a slot address when `lsda_indirect`.
    pub lsda: Option<u64>,
    pub lsda_indirect: bool,
    pub cie: RawCie,
}

/// The bytes of memory from an address to the end of whatever holds it.
pub type Mem<'a> = dyn Fn(u64) -> Option<&'a [u8]> + 'a;

fn parse_cie(mem: &Mem, at: u64, ptr_size: usize) -> Option<RawCie> {
    let b = mem(at)?;
    let mut c = Cursor::new(b, at);
    c.ptr_size = ptr_size;
    let mut len = c.u32()?;
    if len == 0xffff_ffff {
        len = c.u64()?;
    }
    let _ = len;
    if c.u32()? != 0 {
        return None;
    }
    let mut cie = RawCie {
        fde_enc: 0,
        lsda_enc: OMIT,
        personality: None,
    };
    let version = c.u8()?;
    if !matches!(version, 1 | 3 | 4) {
        return None;
    }
    let aug = c.cstr()?;
    if aug.starts_with("eh") {
        // Old gcc "eh" augmentation: a pointer before the alignment factors.
        c.take(ptr_size)?;
    }
    if version >= 4 {
        c.u8()?; // address size
        c.u8()?; // segment size
    }
    c.uleb()?; // code alignment
    c.sleb()?; // data alignment
    if version == 1 {
        c.u8()?;
    } else {
        c.uleb()?;
    }
    if !aug.starts_with('z') {
        return Some(cie);
    }
    c.uleb()?; // augmentation length
    for ch in aug[1..].chars() {
        match ch {
            'L' => cie.lsda_enc = c.u8()?,
            'R' => cie.fde_enc = c.u8()?,
            'P' => {
                let enc = c.u8()?;
                let v = c.encoded(enc)?;
                cie.personality = Some((enc, v));
            }
            _ => {}
        }
    }
    Some(cie)
}

/// The FDE at `at` (a length-prefixed record whose id is not 0), its CIE
/// looked up through `cies`. `None` for a CIE, a terminator or garbage; the
/// second result is the address just after the record.
fn parse_fde(
    mem: &Mem,
    at: u64,
    ptr_size: usize,
    cies: &mut BTreeMap<u64, Option<RawCie>>,
) -> (Option<RawFde>, Option<u64>) {
    let Some(b) = mem(at) else {
        return (None, None);
    };
    let mut c = Cursor::new(b, at);
    c.ptr_size = ptr_size;
    let Some(mut len) = c.u32() else {
        return (None, None);
    };
    if len == 0 {
        return (None, None);
    }
    if len == 0xffff_ffff {
        let Some(l) = c.u64() else {
            return (None, None);
        };
        len = l;
    }
    let body = c.addr();
    let next = body.wrapping_add(len);
    if len as usize > b.len() {
        return (None, None);
    }
    let Some(id) = c.u32() else {
        return (None, Some(next));
    };
    if id == 0 {
        return (None, Some(next));
    }
    let cie_at = body.wrapping_sub(id);
    let cie = cies
        .entry(cie_at)
        .or_insert_with(|| parse_cie(mem, cie_at, ptr_size))
        .clone();
    let Some(cie) = cie else {
        return (None, Some(next));
    };
    let go = |c: &mut Cursor| -> Option<(u64, u64, Option<u64>)> {
        let start = c.encoded(cie.fde_enc)?;
        let range = c.encoded(cie.fde_enc & 0x0f)?;
        let mut lsda = None;
        let aug_len = c.uleb()?;
        let aug_end = c.p.checked_add(usize::try_from(aug_len).ok()?)?;
        if cie.lsda_enc != OMIT {
            lsda = Some(c.encoded(cie.lsda_enc)?);
        }
        c.p = aug_end;
        Some((start, range, lsda))
    };
    match go(&mut c) {
        Some((start, range, lsda)) => (
            Some(RawFde {
                addr: at,
                start,
                end: start.wrapping_add(range),
                lsda,
                lsda_indirect: cie.lsda_enc != OMIT && cie.lsda_enc & 0x80 != 0,
                cie,
            }),
            Some(next),
        ),
        None => (None, Some(next)),
    }
}

/// Walk the CIE/FDE stream from `start` (the `.eh_frame` start) to its
/// terminator: the FDEs, and the address just past the stream.
pub fn walk_stream(mem: &Mem, start: u64, ptr_size: usize) -> (Vec<RawFde>, u64) {
    let mut cies = BTreeMap::new();
    let mut out = Vec::new();
    let mut at = start;
    for _ in 0..1_000_000 {
        let (fde, next) = parse_fde(mem, at, ptr_size, &mut cies);
        if let Some(f) = fde {
            out.push(f);
        }
        match next {
            Some(n) if n > at => at = n,
            _ => break,
        }
    }
    (out, at)
}

/// The header of an `.eh_frame_hdr`: the `.eh_frame` start it names and the
/// search table's `(encoding, count)`, when it has one.
struct Hdr<'a> {
    cursor: Cursor<'a>,
    eh_frame: u64,
    table: Option<(u8, u64)>,
}

fn read_hdr<'a>(mem: &Mem<'a>, hdr: u64, len: u64, ptr_size: usize) -> Option<Hdr<'a>> {
    let b = mem(hdr)?;
    let b = &b[..(len as usize).min(b.len())];
    let mut c = Cursor::new(b, hdr);
    c.datarel = hdr;
    c.ptr_size = ptr_size;
    if c.u8()? != 1 {
        return None;
    }
    let ptr_enc = c.u8()?;
    let count_enc = c.u8()?;
    let table_enc = c.u8()?;
    let eh_frame = c.encoded(ptr_enc)?;
    let count = if count_enc == OMIT {
        None
    } else {
        c.encoded(count_enc)
    };
    let table = match count {
        Some(n) if table_enc != OMIT && enc_size(table_enc, ptr_size) > 0 => Some((table_enc, n)),
        _ => None,
    };
    Some(Hdr {
        cursor: c,
        eh_frame,
        table,
    })
}

/// The FDEs the `.eh_frame_hdr` at `hdr` (of `len` bytes) lists, every one
/// since the unwinder finds them there, and the `.eh_frame` start it names.
/// Without a search table, the stream from that start is walked instead.
pub fn fdes_from_hdr(mem: &Mem, hdr: u64, len: u64, ptr_size: usize) -> Option<(u64, Vec<RawFde>)> {
    let Hdr {
        mut cursor,
        eh_frame,
        table,
    } = read_hdr(mem, hdr, len, ptr_size)?;
    let mut out = Vec::new();
    match table {
        Some((enc, n)) => {
            let mut cies = BTreeMap::new();
            for _ in 0..n {
                let _initial = cursor.encoded(enc)?;
                let fde = cursor.encoded(enc)?;
                if let (Some(f), _) = parse_fde(mem, fde, ptr_size, &mut cies) {
                    out.push(f);
                }
            }
        }
        None => out = walk_stream(mem, eh_frame, ptr_size).0,
    }
    Some((eh_frame, out))
}

/// Function starts from the search table of the `.eh_frame_hdr` at `hdr`,
/// unsorted and possibly with duplicates, without reading a single FDE.
/// `None` when the header is unreadable or has no table.
pub fn hdr_starts(mem: &Mem, hdr: u64, len: u64, ptr_size: usize) -> Option<Vec<u64>> {
    let Hdr {
        mut cursor, table, ..
    } = read_hdr(mem, hdr, len, ptr_size)?;
    let (enc, n) = table?;
    let mut starts = Vec::with_capacity(n.min(1 << 20) as usize);
    for _ in 0..n {
        starts.push(cursor.encoded(enc)?);
        let _fde = cursor.encoded(enc)?;
    }
    Some(starts)
}

/// Function starts from a walk of the `.eh_frame` stream at `start`. A
/// zero-length FDE (a discarded section's remnant) is not a function.
pub fn stream_starts(mem: &Mem, start: u64, ptr_size: usize) -> Vec<u64> {
    walk_stream(mem, start, ptr_size)
        .0
        .into_iter()
        .filter(|f| f.end != f.start)
        .map(|f| f.start)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb128_round_trips() {
        let mut r = Cursor::new(&[0xe5, 0x8e, 0x26, 0x7f, 0x80, 0x7f], 0);
        assert_eq!(r.uleb(), Some(624485));
        assert_eq!(r.sleb(), Some(-1));
        assert_eq!(r.sleb(), Some(-128));
    }

    #[test]
    fn raw_zero_is_a_null_pointer_whatever_the_application() {
        let mut c = Cursor::new(&[0, 0, 0, 0], 0x1000);
        assert_eq!(c.encoded(0x10 | 0x03), Some(0));
    }

    #[test]
    fn absptr_follows_the_pointer_width() {
        let mut c = Cursor::new(&[1, 0, 0, 0, 2, 0, 0, 0], 0);
        c.ptr_size = 4;
        assert_eq!(c.encoded(0x00), Some(1));
        assert_eq!(c.encoded(0x00), Some(2));
    }
}
