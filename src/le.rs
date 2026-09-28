//! Linear Executable (`LE`): the 32-bit container of DOS extenders (DOS/4GW,
//! the Watcom toolchain's default) and of Windows 3.x VxDs.
//!
//! The file is an `MZ` stub (for a DOS/4GW program, the extender itself or a
//! stub that loads it) whose `e_lfanew` points at the `LE` header. Code and
//! data live in *objects*, each with a preferred linear base address and a run
//! of fixed-size pages. The loader is expected to relocate every object, so the
//! page bytes of an absolute reference hold the offset *within its target
//! object*, not a linear address: the fixup records name each such location.
//!
//! [`LeBinary::parse`] maps every object at its preferred base and applies the
//! fixups whose value it can compute (internal references, as offsets and as
//! self-relative displacements), so the bytes it serves read as a flat 32-bit
//! image. It keeps every fixup in [`LeBinary::fixups`] too: they are the exact
//! set of locations that hold an address, and the selector halves (which only
//! the run-time extender can fill) are left for the consumer.
//!
//! Only `LE` is handled. `LX` (OS/2 2.x) shares the header but changes the page
//! map and allows compressed pages; it is reported as [`LeError::Lx`].

use crate::{Arch, BinaryFormat, Endian, Section};

/// Object flag: the object is executable.
pub const OBJ_EXECUTABLE: u32 = 0x0004;
/// Object flag: the object is writable.
pub const OBJ_WRITABLE: u32 = 0x0002;
/// Object flag: the object is readable.
pub const OBJ_READABLE: u32 = 0x0001;
/// Object flag ("big/default"): a 32-bit code object (USE32), or a data object
/// addressed with 32-bit offsets.
pub const OBJ_BIG: u32 = 0x2000;

/// One object of the image, with the file's pages concatenated and the
/// internal fixups applied.
#[derive(Debug, Clone)]
pub struct LeObject {
    /// The 1-based object number fixup records and the header refer to.
    pub number: u32,
    /// Preferred linear base address (`reloc base addr`).
    pub base: u64,
    /// Size in memory. Bytes past `data.len()` are zero.
    pub virtual_size: u64,
    /// The object-table flags (`OBJ_*`).
    pub flags: u32,
    /// The object's file-backed bytes, fixups applied; at most `virtual_size`.
    pub data: Vec<u8>,
}

impl LeObject {
    pub fn executable(&self) -> bool {
        self.flags & OBJ_EXECUTABLE != 0
    }

    pub fn writable(&self) -> bool {
        self.flags & OBJ_WRITABLE != 0
    }

    /// 32-bit (USE32) rather than 16-bit.
    pub fn big(&self) -> bool {
        self.flags & OBJ_BIG != 0
    }

    fn end(&self) -> u64 {
        self.base + self.virtual_size
    }

    fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.end()
    }
}

/// What a fixup writes at its source location (the low nibble of the record's
/// source byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeFixupKind {
    /// `0x0`: the low byte of the target offset.
    Byte,
    /// `0x2`: the target object's 16-bit selector.
    Selector16,
    /// `0x3`: a 16:16 far pointer (offset, then selector).
    Pointer16_16,
    /// `0x5`: a 16-bit offset.
    Offset16,
    /// `0x6`: a 16:32 far pointer (32-bit offset, then selector).
    Pointer16_32,
    /// `0x7`: a 32-bit offset.
    Offset32,
    /// `0x8`: a 32-bit displacement relative to the end of the field.
    Relative32,
}

impl LeFixupKind {
    fn from_nibble(n: u8) -> Option<Self> {
        Some(match n {
            0x0 => LeFixupKind::Byte,
            0x2 => LeFixupKind::Selector16,
            0x3 => LeFixupKind::Pointer16_16,
            0x5 => LeFixupKind::Offset16,
            0x6 => LeFixupKind::Pointer16_32,
            0x7 => LeFixupKind::Offset32,
            0x8 => LeFixupKind::Relative32,
            _ => return None,
        })
    }

    /// Whether the fixup's value includes the target's selector, which only
    /// the run-time extender knows.
    pub fn has_selector(self) -> bool {
        matches!(
            self,
            LeFixupKind::Selector16 | LeFixupKind::Pointer16_16 | LeFixupKind::Pointer16_32
        )
    }
}

/// What a fixup refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeFixupTarget {
    /// An offset in one of this module's objects. `offset` is 0 for a
    /// [`LeFixupKind::Selector16`] fixup, which names only the object.
    Internal { object: u32, offset: u32 },
    /// An export of an imported module, by ordinal. `module` is 1-based into
    /// [`LeBinary::import_modules`].
    ImportOrdinal {
        module: u16,
        ordinal: u32,
        additive: u32,
    },
    /// An export of an imported module, by name.
    ImportName {
        module: u16,
        name: String,
        additive: u32,
    },
    /// An entry of this module's entry table.
    Entry { ordinal: u16, additive: u32 },
}

/// One fixup: a location of the image that holds an address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeFixup {
    /// Linear address of the first byte the fixup writes.
    pub address: u64,
    pub kind: LeFixupKind,
    pub target: LeFixupTarget,
    /// The record's "fixup to 16:16 alias" flag.
    pub alias: bool,
}

/// A parsed `LE` image.
#[derive(Debug, Clone)]
pub struct LeBinary {
    /// The header's CPU type: 1 = 286, 2 = 386, 3 = 486.
    pub cpu: u16,
    /// The header's target OS field. Watcom's DOS/4GW links write 1 (OS/2);
    /// the container does not say "DOS", the stub that loads it does.
    pub os_type: u16,
    /// The header's module flags.
    pub module_flags: u32,
    pub page_size: u32,
    /// The module name: the first entry of the resident name table.
    pub module_name: Option<String>,
    /// Import module names, in the order fixup records number them (1-based).
    pub import_modules: Vec<String>,
    /// Objects in object-table order (`objects[n - 1]` is object `n`).
    pub objects: Vec<LeObject>,
    /// Every fixup record, sources in ascending address order.
    pub fixups: Vec<LeFixup>,
    /// Linear address of the initial `EIP`.
    pub entrypoint: Option<u64>,
    /// Linear address of the initial `ESP`.
    pub initial_stack: Option<u64>,
}

/// Why an `LE` image could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeError {
    /// No `MZ` stub, or `e_lfanew` does not point at an `LE` signature.
    NotLe,
    /// The header is `LX`, which this parser does not handle.
    Lx,
    /// The header declares a big-endian byte or word order.
    BigEndian,
    /// A table or page lies past the end of the file.
    Truncated(&'static str),
    /// A page-map entry is iterated (`EXEPACK`) or of an unknown type.
    UnsupportedPage { page: u32, flags: u8 },
    /// A fixup record has a source type or target type outside the spec.
    BadFixup { offset: usize },
    /// A header or fixup names an object the object table does not have.
    BadObject(u32),
}

impl std::fmt::Display for LeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeError::NotLe => write!(f, "not an LE executable"),
            LeError::Lx => write!(f, "LX executables are not supported"),
            LeError::BigEndian => write!(f, "big-endian LE executables are not supported"),
            LeError::Truncated(what) => write!(f, "LE {what} lies past the end of the file"),
            LeError::UnsupportedPage { page, flags } => {
                write!(f, "LE page {page} has unsupported type {flags}")
            }
            LeError::BadFixup { offset } => {
                write!(f, "malformed LE fixup record at file offset 0x{offset:x}")
            }
            LeError::BadObject(n) => write!(f, "LE refers to missing object {n}"),
        }
    }
}

impl std::error::Error for LeError {}

/// The file offset of the `LE`/`LX` header, if `bytes` is an `MZ` stub whose
/// `e_lfanew` points at one.
pub(crate) fn header_offset(bytes: &[u8]) -> Option<(usize, [u8; 2])> {
    if !bytes.starts_with(b"MZ") {
        return None;
    }
    let at = u32::from_le_bytes(bytes.get(0x3c..0x40)?.try_into().ok()?) as usize;
    let sig: [u8; 2] = bytes.get(at..at + 2)?.try_into().ok()?;
    matches!(&sig, b"LE" | b"LX").then_some((at, sig))
}

/// Little-endian reads that fail with the name of what was being read.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn u8(&self, at: usize, what: &'static str) -> Result<u8, LeError> {
        self.bytes.get(at).copied().ok_or(LeError::Truncated(what))
    }

    fn u16(&self, at: usize, what: &'static str) -> Result<u16, LeError> {
        let b = self.bytes.get(at..at + 2).ok_or(LeError::Truncated(what))?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&self, at: usize, what: &'static str) -> Result<u32, LeError> {
        let b = self.bytes.get(at..at + 4).ok_or(LeError::Truncated(what))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A length-prefixed string, as the name tables store them.
    fn pstr(&self, at: usize, what: &'static str) -> Result<String, LeError> {
        let len = self.u8(at, what)? as usize;
        let b = self
            .bytes
            .get(at + 1..at + 1 + len)
            .ok_or(LeError::Truncated(what))?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
}

impl LeBinary {
    pub fn parse(bytes: &[u8]) -> Result<Self, LeError> {
        let (le, sig) = header_offset(bytes).ok_or(LeError::NotLe)?;
        if &sig == b"LX" {
            return Err(LeError::Lx);
        }
        let r = Reader { bytes };
        let h8 = |off: usize| r.u8(le + off, "header");
        let h16 = |off: usize| r.u16(le + off, "header");
        let h32 = |off: usize| r.u32(le + off, "header");
        if h8(0x02)? != 0 || h8(0x03)? != 0 {
            return Err(LeError::BigEndian);
        }

        let module_pages = h32(0x14)?;
        let page_size = h32(0x28)?;
        let last_page_size = h32(0x2c)?;
        let object_table = le + h32(0x40)? as usize;
        let object_count = h32(0x44)?;
        let page_map = le + h32(0x48)? as usize;
        let resident_names = h32(0x58)?;
        let fixup_pages = le + h32(0x68)? as usize;
        let fixup_records = le + h32(0x6c)? as usize;
        let import_module_table = le + h32(0x70)? as usize;
        let import_module_count = h32(0x74)?;
        let import_procs = le + h32(0x78)? as usize;
        // The one offset in the header that counts from the start of the file.
        let data_pages = h32(0x80)? as usize;

        let module_name = if resident_names != 0 {
            Some(r.pstr(le + resident_names as usize, "resident name table")?)
                .filter(|n| !n.is_empty())
        } else {
            None
        };

        let mut import_modules = Vec::new();
        let mut at = import_module_table;
        for _ in 0..import_module_count {
            let name = r.pstr(at, "import module table")?;
            at += 1 + name.len();
            import_modules.push(name);
        }

        // Objects, and for each object page the logical page number that
        // indexes the fixup page table.
        let mut objects = Vec::new();
        let mut pages: Vec<(u32, u64)> = Vec::new(); // (logical page, linear address)
        for i in 0..object_count {
            let e = object_table + 24 * i as usize;
            let virtual_size = r.u32(e, "object table")? as u64;
            let base = r.u32(e + 4, "object table")? as u64;
            let flags = r.u32(e + 8, "object table")?;
            let first_page = r.u32(e + 12, "object table")?;
            let page_count = r.u32(e + 16, "object table")?;

            let mut data = Vec::new();
            for j in 0..page_count {
                let logical = first_page + j;
                let pm = page_map + 4 * (logical as usize - 1);
                // LE page-map entry: a 24-bit big-endian page number, then a type.
                let number = (r.u8(pm, "page map")? as u32) << 16
                    | (r.u8(pm + 1, "page map")? as u32) << 8
                    | r.u8(pm + 2, "page map")? as u32;
                let kind = r.u8(pm + 3, "page map")?;
                data.resize((j * page_size) as usize, 0);
                match kind {
                    0 => {
                        let size = if number == module_pages {
                            last_page_size
                        } else {
                            page_size
                        };
                        let start = data_pages + (number as usize - 1) * page_size as usize;
                        let page = bytes
                            .get(start..start + size as usize)
                            .ok_or(LeError::Truncated("page"))?;
                        data.extend_from_slice(page);
                    }
                    // Zero-filled (3) and invalid (2) pages read as zeros.
                    2 | 3 => data.resize(((j + 1) * page_size) as usize, 0),
                    flags => {
                        return Err(LeError::UnsupportedPage {
                            page: logical,
                            flags,
                        });
                    }
                }
                pages.push((logical, base + (j * page_size) as u64));
            }
            data.truncate(virtual_size as usize);
            objects.push(LeObject {
                number: i + 1,
                base,
                virtual_size,
                flags,
                data,
            });
        }

        let mut fixups = Vec::new();
        for &(logical, page_va) in &pages {
            let slot = fixup_pages + 4 * (logical as usize - 1);
            let start = fixup_records + r.u32(slot, "fixup page table")? as usize;
            let end = fixup_records + r.u32(slot + 4, "fixup page table")? as usize;
            let mut at = start;
            while at < end {
                at = parse_fixup(&r, at, page_va, import_procs, &mut fixups)?;
            }
        }
        fixups.sort_by_key(|f| f.address);

        let mut binary = LeBinary {
            cpu: h16(0x08)?,
            os_type: h16(0x0a)?,
            module_flags: h32(0x10)?,
            page_size,
            module_name,
            import_modules,
            objects,
            fixups: Vec::new(),
            entrypoint: None,
            initial_stack: None,
        };
        binary.entrypoint = binary.linear(h32(0x18)?, h32(0x1c)?)?;
        binary.initial_stack = binary.linear(h32(0x20)?, h32(0x24)?)?;
        for f in &fixups {
            binary.apply(f)?;
        }
        binary.fixups = fixups;
        Ok(binary)
    }

    /// The linear address of `offset` in object `object`, or `None` when the
    /// object number is 0 (the header's "none").
    fn linear(&self, object: u32, offset: u32) -> Result<Option<u64>, LeError> {
        if object == 0 {
            return Ok(None);
        }
        let obj = self.object(object)?;
        Ok(Some(obj.base + offset as u64))
    }

    fn object(&self, number: u32) -> Result<&LeObject, LeError> {
        (number as usize)
            .checked_sub(1)
            .and_then(|i| self.objects.get(i))
            .ok_or(LeError::BadObject(number))
    }

    /// The linear address a fixup refers to, when the image alone determines
    /// it: an internal reference. Imports and entry-table references need the
    /// run-time loader.
    pub fn fixup_target(&self, fixup: &LeFixup) -> Option<u64> {
        match fixup.target {
            LeFixupTarget::Internal { object, offset } => {
                self.linear(object, offset).ok().flatten()
            }
            _ => None,
        }
    }

    /// Write the offset part of an internal fixup into its object's bytes.
    /// Selector parts are left as the file has them.
    fn apply(&mut self, fixup: &LeFixup) -> Result<(), LeError> {
        let LeFixupTarget::Internal { object, .. } = fixup.target else {
            return Ok(());
        };
        let target = self.fixup_target(fixup).ok_or(LeError::BadObject(object))?;
        let at = fixup.address;
        let value: Vec<u8> = match fixup.kind {
            LeFixupKind::Selector16 => return Ok(()),
            LeFixupKind::Byte => vec![target as u8],
            LeFixupKind::Offset16 | LeFixupKind::Pointer16_16 => {
                (target as u16).to_le_bytes().to_vec()
            }
            LeFixupKind::Offset32 | LeFixupKind::Pointer16_32 => {
                (target as u32).to_le_bytes().to_vec()
            }
            LeFixupKind::Relative32 => (target.wrapping_sub(at + 4) as u32).to_le_bytes().to_vec(),
        };
        // A fixup may straddle the end of its object's file-backed bytes: the
        // zero-filled tail is part of the object, so `data` grows to hold it
        // (up to the virtual size). RUN.EXE's last fini record's pointer does.
        if let Some(obj) = self.objects.iter_mut().find(|o| o.contains(at)) {
            let off = (at - obj.base) as usize;
            let end = (off + value.len()).min(obj.virtual_size as usize);
            if obj.data.len() < end {
                obj.data.resize(end, 0);
            }
            for (i, b) in value.into_iter().enumerate() {
                if let Some(slot) = obj.data.get_mut(off + i) {
                    *slot = b;
                }
            }
        }
        Ok(())
    }

    fn object_at(&self, addr: u64) -> Option<&LeObject> {
        self.objects.iter().find(|o| o.contains(addr))
    }
}

/// Parse the fixup record at file offset `at`, whose source offsets are
/// relative to the page at `page_va`; push one [`LeFixup`] per source and
/// return the offset of the next record.
fn parse_fixup(
    r: &Reader<'_>,
    at: usize,
    page_va: u64,
    import_procs: usize,
    out: &mut Vec<LeFixup>,
) -> Result<usize, LeError> {
    let bad = LeError::BadFixup { offset: at };
    let src = r.u8(at, "fixup record")?;
    let flags = r.u8(at + 1, "fixup record")?;
    let kind = LeFixupKind::from_nibble(src & 0x0f).ok_or(bad.clone())?;
    let alias = src & 0x10 != 0;
    let source_list = src & 0x20 != 0;
    let mut p = at + 2;

    // The source offset (signed: a fixup may start on the previous page), or
    // the count of offsets listed after the target.
    let (first, count) = if source_list {
        let n = r.u8(p, "fixup record")?;
        p += 1;
        (None, n)
    } else {
        let off = r.u16(p, "fixup record")? as i16;
        p += 2;
        (Some(off), 1)
    };

    let read_object = |p: &mut usize| -> Result<u16, LeError> {
        if flags & 0x40 != 0 {
            let v = r.u16(*p, "fixup record")?;
            *p += 2;
            Ok(v)
        } else {
            let v = r.u8(*p, "fixup record")? as u16;
            *p += 1;
            Ok(v)
        }
    };
    let read_additive = |p: &mut usize| -> Result<u32, LeError> {
        if flags & 0x04 == 0 {
            return Ok(0);
        }
        if flags & 0x20 != 0 {
            let v = r.u32(*p, "fixup record")?;
            *p += 4;
            Ok(v)
        } else {
            let v = r.u16(*p, "fixup record")? as u32;
            *p += 2;
            Ok(v)
        }
    };
    let read_offset = |p: &mut usize| -> Result<u32, LeError> {
        if flags & 0x10 != 0 {
            let v = r.u32(*p, "fixup record")?;
            *p += 4;
            Ok(v)
        } else {
            let v = r.u16(*p, "fixup record")? as u32;
            *p += 2;
            Ok(v)
        }
    };

    let target = match flags & 0x03 {
        0 => {
            let object = read_object(&mut p)? as u32;
            let offset = if kind == LeFixupKind::Selector16 {
                0
            } else {
                read_offset(&mut p)?
            };
            LeFixupTarget::Internal { object, offset }
        }
        1 => {
            let module = read_object(&mut p)?;
            let ordinal = if flags & 0x80 != 0 {
                let v = r.u8(p, "fixup record")? as u32;
                p += 1;
                v
            } else {
                read_offset(&mut p)?
            };
            let additive = read_additive(&mut p)?;
            LeFixupTarget::ImportOrdinal {
                module,
                ordinal,
                additive,
            }
        }
        2 => {
            let module = read_object(&mut p)?;
            let name_at = read_offset(&mut p)? as usize;
            let name = r.pstr(import_procs + name_at, "import procedure name table")?;
            let additive = read_additive(&mut p)?;
            LeFixupTarget::ImportName {
                module,
                name,
                additive,
            }
        }
        _ => {
            let ordinal = read_object(&mut p)?;
            let additive = read_additive(&mut p)?;
            LeFixupTarget::Entry { ordinal, additive }
        }
    };

    let mut push = |off: i16| {
        out.push(LeFixup {
            address: page_va.wrapping_add_signed(off as i64),
            kind,
            target: target.clone(),
            alias,
        })
    };
    match first {
        Some(off) => push(off),
        None => {
            for _ in 0..count {
                push(r.u16(p, "fixup record")? as i16);
                p += 2;
            }
        }
    }
    Ok(p)
}

impl BinaryFormat for LeBinary {
    /// The lowest object base.
    fn load_address(&self) -> u64 {
        self.objects.iter().map(|o| o.base).min().unwrap_or(0)
    }

    fn architecture(&self) -> Arch {
        Arch::I386
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }

    /// 32 when the entry point's object (or, without one, any code object) is
    /// USE32; 16 otherwise.
    fn bits(&self) -> u32 {
        let code = match self.entrypoint.and_then(|e| self.object_at(e)) {
            Some(o) => o.big(),
            None => self.objects.iter().any(|o| o.executable() && o.big()),
        };
        if code { 32 } else { 16 }
    }

    /// Objects have no names; each is reported as `object<N>`.
    fn sections(&self) -> Vec<Section> {
        self.objects
            .iter()
            .map(|o| Section {
                name: format!("object{}", o.number),
                address: o.base,
                size: o.virtual_size,
                writable: o.writable(),
                executable: o.executable(),
            })
            .collect()
    }

    fn linked_libraries(&self) -> Vec<String> {
        self.import_modules.clone()
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        let o = self.object_at(addr)?;
        Some(o.data.get((addr - o.base) as usize).copied().unwrap_or(0))
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        let o = self.object_at(addr)?;
        o.data.get((addr - o.base) as usize..)
    }

    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        self.objects
            .iter()
            .map(|o| {
                let mut bytes = o.data.clone();
                bytes.resize(o.virtual_size as usize, 0);
                (o.base, bytes, o.executable(), o.writable())
            })
            .collect()
    }

    fn is_executable(&self, addr: u64) -> bool {
        self.object_at(addr).is_some_and(|o| o.executable())
    }

    fn is_known_writable(&self, addr: u64) -> bool {
        self.object_at(addr).is_some_and(|o| o.writable())
    }

    /// DOS extenders do not enforce the read-only flag (DOS/4GW maps every
    /// object read-write), so an object without it is not proven immutable.
    fn is_known_read_only(&self, _addr: u64) -> bool {
        false
    }

    fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.object_at(addr).map(|o| (o.base, o.end()))
    }

    fn entry_points(&self) -> Vec<u64> {
        self.entrypoint.into_iter().collect()
    }

    fn entrypoint(&self) -> Option<u64> {
        self.entrypoint
    }

    fn symbol_name(&self, addr: u64) -> Option<&str> {
        (Some(addr) == self.entrypoint).then_some("_start")
    }

    fn symbol_address(&self, name: &str) -> Option<u64> {
        (name == "_start").then_some(self.entrypoint).flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal LE: a 32-bit code object at 0x10000 and a data object at
    /// 0x20000, one page each, with fixups:
    /// - code+1: `mov eax, [data+0x10]` → Offset32 to object 2 offset 0x10
    /// - code+6: `call code+0x20`       → Relative32 to object 1 offset 0x20
    /// - code+0xb: far pointer          → Pointer16_32 to object 2 offset 4
    /// - data+0, data+8 and data+0x1e (a source list; the last straddles the
    ///   end of the file-backed bytes) → Offset32 to object 1 offset 0x20
    pub(crate) fn sample() -> Vec<u8> {
        let le = 0x40usize;
        let page = 0x200u32;
        let mut f = vec![0u8; 0x600];
        f[..2].copy_from_slice(b"MZ");
        f[0x3c..0x40].copy_from_slice(&(le as u32).to_le_bytes());
        let put32 =
            |f: &mut Vec<u8>, at: usize, v: u32| f[at..at + 4].copy_from_slice(&v.to_le_bytes());
        f[le..le + 2].copy_from_slice(b"LE");
        f[le + 0x08] = 2; // 386
        f[le + 0x0a] = 1;
        put32(&mut f, le + 0x14, 2); // module pages
        put32(&mut f, le + 0x18, 1); // eip object
        put32(&mut f, le + 0x1c, 0x0); // eip
        put32(&mut f, le + 0x20, 2); // esp object
        put32(&mut f, le + 0x24, 0x100); // esp
        put32(&mut f, le + 0x28, page);
        put32(&mut f, le + 0x2c, 0x20); // last page size
        put32(&mut f, le + 0x40, 0xc4); // object table
        put32(&mut f, le + 0x44, 2);
        put32(&mut f, le + 0x48, 0xf4); // page map
        put32(&mut f, le + 0x58, 0xfc); // resident names
        put32(&mut f, le + 0x68, 0x104); // fixup page table
        put32(&mut f, le + 0x6c, 0x110); // fixup records
        put32(&mut f, le + 0x70, 0x160); // import modules
        put32(&mut f, le + 0x78, 0x160); // import procs
        put32(&mut f, le + 0x80, 0x200); // data pages (file offset)

        // Objects: code (0x40 bytes, RX, big), data (0x100 bytes of which 0x20 in file, RW).
        let ot = le + 0xc4;
        for (i, (vs, base, flags, first)) in [
            (
                0x40u32,
                0x10000u32,
                OBJ_READABLE | OBJ_EXECUTABLE | OBJ_BIG,
                1u32,
            ),
            (0x100, 0x20000, OBJ_READABLE | OBJ_WRITABLE | OBJ_BIG, 2),
        ]
        .into_iter()
        .enumerate()
        {
            let e = ot + 24 * i;
            put32(&mut f, e, vs);
            put32(&mut f, e + 4, base);
            put32(&mut f, e + 8, flags);
            put32(&mut f, e + 12, first);
            put32(&mut f, e + 16, 1);
        }
        // Page map: logical 1 -> physical 1, logical 2 -> physical 2.
        f[le + 0xf4..le + 0xf8].copy_from_slice(&[0, 0, 1, 0]);
        f[le + 0xf8..le + 0xfc].copy_from_slice(&[0, 0, 2, 0]);
        f[le + 0xfc] = 4;
        f[le + 0xfd..le + 0x101].copy_from_slice(b"demo");

        // Fixup records: page 1's, then page 2's.
        let recs: Vec<u8> = [
            // Offset32, internal, object 2, 16-bit offset 0x10, at +1
            vec![0x07, 0x00, 0x01, 0x00, 0x02, 0x10, 0x00],
            // Relative32, internal, object 1, offset 0x20, at +6
            vec![0x08, 0x00, 0x06, 0x00, 0x01, 0x20, 0x00],
            // Pointer16_32, internal, object 2, offset 4, at +0xb
            vec![0x06, 0x00, 0x0b, 0x00, 0x02, 0x04, 0x00],
        ]
        .concat();
        let page2: Vec<u8> = vec![
            // Offset32, source list of 3, object 1, 32-bit offset 0x20; sources 0, 8, 0x1e
            0x27, 0x10, 0x03, 0x01, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x1e, 0x00,
        ];
        let fr = le + 0x110;
        f[fr..fr + recs.len()].copy_from_slice(&recs);
        f[fr + recs.len()..fr + recs.len() + page2.len()].copy_from_slice(&page2);
        put32(&mut f, le + 0x104, 0);
        put32(&mut f, le + 0x108, recs.len() as u32);
        put32(&mut f, le + 0x10c, (recs.len() + page2.len()) as u32);

        // Page 1: code with the object-relative values the linker leaves.
        let code: [u8; 17] = [
            0xa1, 0x10, 0, 0, 0, // mov eax, [0x10]
            0xe8, 0, 0, 0, 0, // call rel32
            0x90, 0, 0, 0, 0, 0, 0, // far pointer slot
        ];
        f[0x200..0x200 + code.len()].copy_from_slice(&code);
        f[0x200 + page as usize..0x200 + page as usize + 0x20].fill(0xcc);
        f.truncate(0x200 + page as usize + 0x20);
        f
    }

    #[test]
    fn parses_objects_and_applies_internal_fixups() {
        let b = LeBinary::parse(&sample()).unwrap();
        assert_eq!(b.module_name.as_deref(), Some("demo"));
        assert_eq!(b.objects.len(), 2);
        assert_eq!(b.entrypoint, Some(0x10000));
        assert_eq!(b.initial_stack, Some(0x20100));
        assert_eq!(b.bits(), 32);
        assert_eq!(b.architecture(), Arch::I386);

        assert_eq!(b.read_uint(0x10001, 4), Some(0x20010));
        // call at 0x10005: rel32 at 0x10006, target 0x10020.
        assert_eq!(b.read_uint(0x10006, 4), Some(0x10020 - 0x1000a));
        assert_eq!(b.read_uint(0x1000b, 4), Some(0x20004));
        assert_eq!(b.read_uint(0x20000, 4), Some(0x10020));
        assert_eq!(b.read_uint(0x20008, 4), Some(0x10020));
        assert_eq!(b.read_uint(0x2001e, 4), Some(0x10020));
        // Unfixed data bytes and the zero-filled tail.
        assert_eq!(b.byte_at(0x20004), Some(0xcc));
        assert_eq!(b.byte_at(0x200ff), Some(0));
        assert_eq!(b.byte_at(0x20100), None);
    }

    #[test]
    fn keeps_every_fixup_with_its_target() {
        let b = LeBinary::parse(&sample()).unwrap();
        let rows: Vec<_> = b
            .fixups
            .iter()
            .map(|f| (f.address, f.kind, b.fixup_target(f)))
            .collect();
        assert_eq!(
            rows,
            vec![
                (0x10001, LeFixupKind::Offset32, Some(0x20010)),
                (0x10006, LeFixupKind::Relative32, Some(0x10020)),
                (0x1000b, LeFixupKind::Pointer16_32, Some(0x20004)),
                (0x20000, LeFixupKind::Offset32, Some(0x10020)),
                (0x20008, LeFixupKind::Offset32, Some(0x10020)),
                (0x2001e, LeFixupKind::Offset32, Some(0x10020)),
            ]
        );
        assert!(LeFixupKind::Pointer16_32.has_selector());
    }

    #[test]
    fn reports_object_permissions() {
        let b = LeBinary::parse(&sample()).unwrap();
        assert!(b.is_executable(0x10000));
        assert!(!b.is_executable(0x20000));
        assert!(b.is_known_writable(0x20000));
        assert!(!b.is_known_read_only(0x10000));
        let names: Vec<_> = b.sections().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["object1", "object2"]);
    }

    #[test]
    fn rejects_lx_and_bare_mz() {
        let mut f = sample();
        f[0x40..0x42].copy_from_slice(b"LX");
        assert_eq!(LeBinary::parse(&f).unwrap_err(), LeError::Lx);
        f[0x40..0x42].copy_from_slice(b"NE");
        assert_eq!(LeBinary::parse(&f).unwrap_err(), LeError::NotLe);
    }
}
