//! A borrowed, loader-level view of an ELF file: [`Image`].
//!
//! Everything here is what the kernel and `ld.so` read to start the program,
//! plus the section names a linker leaves behind. One pass over the file's
//! headers builds it; nothing is copied out of the input (names and segment
//! bytes borrow from it) and **`.eh_frame` is never parsed at load**: the
//! exception tables, and the function starts they list, are read on demand by
//! [`Image::eh`] and [`Image::seeds`].
//!
//! The view is the program headers and the dynamic table first (`PT_LOAD`,
//! `PT_DYNAMIC`, `PT_TLS`, `PT_GNU_RELRO`): a file stripped of its section
//! headers (`objcopy --strip-section-headers`) reads the same, minus the
//! section names, so [`Image::sections`] is then empty and no name is
//! invented. When the section headers are there they add what the dynamic
//! table cannot state: the section names, `.symtab`, the constructor arrays
//! and relocation tables of a static executable (which has no `.dynamic`).
//!
//! Little-endian ELF only (32- or 64-bit); a big-endian file is an
//! [`ElfError::Unsupported`].

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::LoadError;
use crate::eh::Eh;
use crate::elf::{ElfError, ElfKind};

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;
const PT_NOTE: u32 = 4;
const PT_PHDR: u32 = 6;
const PT_TLS: u32 = 7;
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
const PT_GNU_RELRO: u32 = 0x6474_e552;

const SHT_SYMTAB: u32 = 2;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;
const SHT_DYNSYM: u32 = 11;
const SHF_ALLOC: u64 = 2;

const DT_NEEDED: i64 = 1;
const DT_PLTRELSZ: i64 = 2;
const DT_HASH: i64 = 4;
const DT_PLTGOT: i64 = 3;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_INIT: i64 = 12;
const DT_FINI: i64 = 13;
const DT_REL: i64 = 17;
const DT_RELSZ: i64 = 18;
const DT_TEXTREL: i64 = 22;
const DT_JMPREL: i64 = 23;
const DT_INIT_ARRAY: i64 = 25;
const DT_FINI_ARRAY: i64 = 26;
const DT_INIT_ARRAYSZ: i64 = 27;
const DT_FINI_ARRAYSZ: i64 = 28;
const DT_FLAGS: i64 = 30;
const DT_PREINIT_ARRAY: i64 = 32;
const DT_PREINIT_ARRAYSZ: i64 = 33;
const DT_RELRSZ: i64 = 35;
const DT_RELR: i64 = 36;
const DT_GNU_HASH: i64 = 0x6fff_fef5;
const DT_VERSYM: i64 = 0x6fff_fff0;
const DT_VERNEED: i64 = 0x6fff_fffe;
const DT_VERNEEDNUM: i64 = 0x6fff_ffff;
const DF_TEXTREL: u64 = 4;

const EM_386: u16 = 3;
const EM_X86_64: u16 = 62;
const SHN_UNDEF: u16 = 0;

/// A `PT_LOAD` segment: what the loader maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Segment<'a> {
    pub vaddr: u64,
    /// Size in memory; the tail past `bytes` is zero-filled.
    pub memsz: u64,
    /// Offset of `bytes` in the file.
    pub file_off: u64,
    /// `p_flags`: X = 1, W = 2, R = 4.
    pub flags: u32,
    /// The file-backed bytes (`p_filesz` of them).
    pub bytes: &'a [u8],
    /// `p_align`; zero or one means none.
    pub align: u64,
}

impl Segment<'_> {
    pub fn executable(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn writable(&self) -> bool {
        self.flags & 2 != 0
    }

    /// Whether `va` is inside the segment's memory image (file bytes and
    /// zero-filled tail).
    pub fn contains(&self, va: u64) -> bool {
        va >= self.vaddr && va - self.vaddr < self.memsz
    }
}

/// An allocated section header (`SHF_ALLOC`). The list is empty when the file
/// has no section headers, or when they were not asked for
/// ([`Image::load_segments`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Section<'a> {
    pub name: &'a str,
    pub addr: u64,
    pub size: u64,
    /// `sh_flags`: `SHF_WRITE` = 1, `SHF_ALLOC` = 2, `SHF_EXECINSTR` = 4,
    /// `SHF_TLS` = 0x400.
    pub flags: u64,
    /// `SHT_NOBITS`: no file bytes (`.bss`, `.tbss`).
    pub nobits: bool,
    /// The section's file bytes; empty for a NOBITS section or one whose
    /// range lies outside the file.
    pub bytes: &'a [u8],
}

impl Section<'_> {
    pub fn executable(&self) -> bool {
        self.flags & 4 != 0
    }

    pub fn writable(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn tls(&self) -> bool {
        self.flags & 0x400 != 0
    }
}

/// What a dynamic relocation computes, for the machines this crate knows
/// (x86-64 and i386); anything else, and the types it has no name for, is
/// [`RelocKind::Other`] with the raw `r_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RelocKind {
    /// `R_*_RELATIVE`: the load bias plus the addend.
    Relative,
    /// `R_X86_64_64` / `R_386_32`: symbol plus addend.
    Abs,
    /// `R_*_GLOB_DAT`: a GOT slot of a symbol.
    GlobDat,
    /// `R_*_JUMP_SLOT`: a PLT's GOT slot.
    JumpSlot,
    /// `R_*_COPY`: the symbol's initial value copied into the executable.
    Copy,
    /// `R_X86_64_TPOFF64` / `R_386_TLS_TPOFF`: the thread-pointer offset of a
    /// TLS variable.
    TpOff,
    /// `R_*_IRELATIVE`: the loader calls the addend as a resolver.
    IRelative,
    Other(u32),
}

/// One dynamic relocation entry (`DT_RELA`, `DT_JMPREL`, `DT_REL`, or an
/// allocated relocation section of a file with no dynamic table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Reloc {
    /// `r_offset`: the location the loader writes.
    pub offset: u64,
    pub kind: RelocKind,
    /// The `.dynsym` index of the symbol; 0 for none.
    pub sym: u32,
    /// `r_addend` (0 for a REL entry, whose addend lives in the place).
    pub addend: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymbolKind {
    /// `STT_FUNC` or `STT_GNU_IFUNC`.
    Func,
    Object,
    Tls,
    Other,
}

/// Which table a [`Symbol`] was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Table {
    Symtab,
    Dynsym,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Symbol<'a> {
    pub name: &'a str,
    /// `st_value`.
    pub addr: u64,
    pub size: u64,
    pub kind: SymbolKind,
    /// `st_shndx` is not `SHN_UNDEF`.
    pub defined: bool,
    pub table: Table,
}

/// An imported function: the GOT slot the loader fills, the PLT stub that
/// jumps through it, and the symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Import<'a> {
    /// The GOT slot (`R_*_JUMP_SLOT` / `R_*_GLOB_DAT` offset); `None` for a
    /// canonical PLT entry known only from an undefined `.dynsym` value (an
    /// address-taken import).
    pub slot: Option<u64>,
    /// A PLT stub (`.plt`, `.plt.sec`, `.plt.got`) found by scanning
    /// executable memory for `jmp *slot(%rip)`. Several stubs may jump through
    /// one slot; each is an import of its own.
    pub stub: Option<u64>,
    /// Which kind of stub that is; `None` when there is no stub, or when the
    /// address is a canonical PLT entry (known from `.dynsym` alone).
    pub stub_kind: Option<StubKind>,
    pub name: &'a str,
    /// The soname the symbol's version requirement (`.gnu.version_r`) names.
    pub library: Option<&'a str>,
}

/// The kind of a PLT stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StubKind {
    /// `.plt` / `.plt.sec`: lazy binding through a `JUMP_SLOT`.
    Plt,
    /// `.plt.got`: the stub of an import whose GOT slot is a `GLOB_DAT`
    /// (a PIE's `__cxa_finalize`, anything under `-z now` that is also
    /// address-taken).
    PltGot,
}

/// The dynamic table, as far as a loader reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Dynamic<'a> {
    /// `DT_NEEDED` sonames, in order.
    pub needed: Vec<&'a str>,
    /// `DT_INIT` / `DT_FINI` (when non-zero).
    pub init: Option<u64>,
    pub fini: Option<u64>,
    /// `(address, size in bytes)` of the constructor / destructor arrays: the
    /// dynamic tags', else (a static executable has no `.dynamic`) the
    /// section of that name.
    pub init_array: Option<(u64, u64)>,
    pub fini_array: Option<(u64, u64)>,
    pub preinit_array: Option<(u64, u64)>,
    /// `DT_PLTGOT`: the address of the PLT's GOT (`.got.plt`).
    pub pltgot: Option<u64>,
    /// Where the table itself is: `PT_DYNAMIC`'s `(vaddr, memsz)`.
    pub table: Option<(u64, u64)>,
    /// `DT_TEXTREL`, or `DT_FLAGS` with `DF_TEXTREL`: the code holds
    /// absolute addresses the loader patches.
    pub textrel: bool,
}

/// The `PT_TLS` template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Tls<'a> {
    pub vaddr: u64,
    pub memsz: u64,
    pub align: u64,
    /// `.tdata`: the first `p_filesz` bytes of the block (the rest is `.tbss`,
    /// zeros).
    pub init: &'a [u8],
}

/// A function start the file's own metadata names: not a symbol, but a place
/// the unwinder or the loader will reach. See [`Image::seeds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Seed {
    pub addr: u64,
    pub source: SeedSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeedSource {
    /// An FDE's `initial_location`.
    EhFrame,
    /// `DT_INIT` / `DT_FINI` or an entry of a constructor / destructor array.
    InitFini,
}

#[derive(Debug, Clone, Copy)]
struct Phdr {
    kind: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

struct Shdr {
    name: u32,
    kind: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
}

/// The little-endian field readers, by ELF class.
#[derive(Clone, Copy)]
struct Fmt {
    is_64: bool,
}

impl Fmt {
    fn word(self) -> usize {
        if self.is_64 { 8 } else { 4 }
    }

    fn sym_size(self) -> usize {
        if self.is_64 { 24 } else { 16 }
    }

    fn rela_size(self) -> usize {
        if self.is_64 { 24 } else { 12 }
    }

    fn rel_size(self) -> usize {
        if self.is_64 { 16 } else { 8 }
    }

    fn dyn_size(self) -> usize {
        if self.is_64 { 16 } else { 8 }
    }

    fn uword(self, b: &[u8], at: usize) -> Option<u64> {
        if self.is_64 {
            le64(b, at)
        } else {
            le32(b, at).map(u64::from)
        }
    }
}

fn le16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        b.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn le32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn le64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        b.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// The NUL-terminated string at `at`, or `""` when it is not UTF-8.
fn cstr(b: &[u8], at: usize) -> &str {
    let Some(tail) = b.get(at..) else { return "" };
    let end = tail.iter().position(|&c| c == 0).unwrap_or(tail.len());
    std::str::from_utf8(&tail[..end]).unwrap_or("")
}

/// Options for [`Image::load_with`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LoadOptions {
    /// Read the section headers (names, `.symtab`, section-sized tables).
    /// When `false` the file is read as the loader reads it, through the
    /// program headers and the dynamic table only. Default `true`.
    pub sections: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { sections: true }
    }
}

/// A loaded ELF file. See the [module docs](self).
pub struct Image<'a> {
    file: &'a [u8],
    fmt: Fmt,
    pub kind: ElfKind,
    /// `e_machine`.
    pub machine: u16,
    pub entry: u64,
    /// `PT_LOAD` segments, by address.
    pub segments: Vec<Segment<'a>>,
    /// The allocated sections, in header order; empty when there are no
    /// section headers.
    pub sections: Vec<Section<'a>>,
    /// Dynamic relocations: `DT_RELA`, then `DT_JMPREL`, then `DT_REL` (in
    /// each table's order), then those of allocated relocation sections no
    /// dynamic tag covers.
    pub relocs: Vec<Reloc>,
    /// `.dynsym` (entry 0, the null symbol, included, so an index is a
    /// [`Reloc::sym`]) then `.symtab`; see [`Symbol::table`].
    pub symbols: Vec<Symbol<'a>>,
    /// Imports in relocation order, then the canonical PLT entries of
    /// address-taken imports.
    pub imports: Vec<Import<'a>>,
    pub dynamic: Dynamic<'a>,
    pub tls: Option<Tls<'a>>,
    /// `PT_GNU_RELRO`: `(vaddr, memsz)`.
    pub relro: Option<(u64, u64)>,
    phdrs: Vec<Phdr>,
    dynsym_count: usize,
    /// `(other, soname)` of every `Vernaux`.
    version_needs: Vec<(u16, &'a str)>,
    versym: Option<u64>,
    /// `.eh_frame_hdr`'s `(address, size)`: `PT_GNU_EH_FRAME`, else the
    /// section.
    eh_hdr: Option<(u64, u64)>,
    import_names: HashMap<u64, &'a str>,
    reloc_index: OnceLock<HashMap<u64, usize>>,
    eh: OnceLock<Eh>,
}

/// Load an ELF file (the only format [`Image`] reads yet) as an [`Image`].
pub fn load(bytes: &[u8]) -> Result<Image<'_>, LoadError> {
    Image::load(bytes)
}

impl<'a> Image<'a> {
    /// Load `bytes` as an ELF file.
    pub fn load(bytes: &'a [u8]) -> Result<Self, LoadError> {
        Self::load_with(bytes, LoadOptions::default())
    }

    /// Load `bytes` through the program headers and the dynamic table only;
    /// the section headers are not read at all ([`Image::sections`] is
    /// empty).
    pub fn load_segments(bytes: &'a [u8]) -> Result<Self, LoadError> {
        Self::load_with(bytes, LoadOptions { sections: false })
    }

    pub fn load_with(bytes: &'a [u8], opts: LoadOptions) -> Result<Self, LoadError> {
        match crate::Format::detect(bytes) {
            Some(crate::Format::Elf) => {}
            Some(other) => return Err(LoadError::Unsupported(other)),
            None => return Err(LoadError::UnknownFormat),
        }
        Self::parse(bytes, opts).map_err(LoadError::Elf)
    }

    fn parse(file: &'a [u8], opts: LoadOptions) -> Result<Self, ElfError> {
        let trunc = || ElfError::Truncated("ELF header");
        let class = *file.get(4).ok_or_else(trunc)?;
        let fmt = match class {
            1 => Fmt { is_64: false },
            2 => Fmt { is_64: true },
            _ => return Err(ElfError::Unsupported("ELF class")),
        };
        if file.get(5) != Some(&1) {
            return Err(ElfError::Unsupported("big-endian ELF"));
        }
        let kind = ElfKind::from_e_type(le16(file, 16).ok_or_else(trunc)?);
        let machine = le16(file, 18).ok_or_else(trunc)?;
        let (entry, phoff, shoff, phentsize, phnum, shentsize, shnum, shstrndx) = if fmt.is_64 {
            (
                le64(file, 24).ok_or_else(trunc)?,
                le64(file, 32).ok_or_else(trunc)?,
                le64(file, 40).ok_or_else(trunc)?,
                le16(file, 54).ok_or_else(trunc)? as usize,
                le16(file, 56).ok_or_else(trunc)? as usize,
                le16(file, 58).ok_or_else(trunc)? as usize,
                le16(file, 60).ok_or_else(trunc)? as usize,
                le16(file, 62).ok_or_else(trunc)? as usize,
            )
        } else {
            (
                u64::from(le32(file, 24).ok_or_else(trunc)?),
                u64::from(le32(file, 28).ok_or_else(trunc)?),
                u64::from(le32(file, 32).ok_or_else(trunc)?),
                le16(file, 42).ok_or_else(trunc)? as usize,
                le16(file, 44).ok_or_else(trunc)? as usize,
                le16(file, 46).ok_or_else(trunc)? as usize,
                le16(file, 48).ok_or_else(trunc)? as usize,
                le16(file, 50).ok_or_else(trunc)? as usize,
            )
        };

        // --- program headers -------------------------------------------------
        let mut phdrs = Vec::with_capacity(phnum);
        for i in 0..phnum {
            let at = (phoff as usize)
                .checked_add(i * phentsize)
                .ok_or(ElfError::Truncated("program headers"))?;
            let ph = if fmt.is_64 {
                (|| {
                    Some(Phdr {
                        kind: le32(file, at)?,
                        flags: le32(file, at + 4)?,
                        offset: le64(file, at + 8)?,
                        vaddr: le64(file, at + 16)?,
                        filesz: le64(file, at + 32)?,
                        memsz: le64(file, at + 40)?,
                        align: le64(file, at + 48)?,
                    })
                })()
            } else {
                (|| {
                    Some(Phdr {
                        kind: le32(file, at)?,
                        offset: u64::from(le32(file, at + 4)?),
                        vaddr: u64::from(le32(file, at + 8)?),
                        filesz: u64::from(le32(file, at + 16)?),
                        memsz: u64::from(le32(file, at + 20)?),
                        flags: le32(file, at + 24)?,
                        align: u64::from(le32(file, at + 28)?),
                    })
                })()
            };
            phdrs.push(ph.ok_or(ElfError::Truncated("program headers"))?);
        }
        let mut segments = Vec::new();
        let mut relro = None;
        let mut dynamic_at = None;
        let mut tls = None;
        let mut eh_hdr = None;
        for ph in &phdrs {
            match ph.kind {
                PT_LOAD => {
                    let start = ph.offset as usize;
                    let end = start.checked_add(ph.filesz as usize);
                    let bytes = end
                        .and_then(|e| file.get(start..e))
                        .ok_or(ElfError::SegmentOutOfBounds(ph.vaddr))?;
                    segments.push(Segment {
                        vaddr: ph.vaddr,
                        memsz: ph.memsz,
                        file_off: ph.offset,
                        flags: ph.flags,
                        bytes,
                        align: ph.align,
                    });
                }
                PT_DYNAMIC => dynamic_at = Some((ph.vaddr, ph.memsz)),
                PT_GNU_RELRO => relro = Some((ph.vaddr, ph.memsz)),
                PT_GNU_EH_FRAME if ph.memsz > 0 => eh_hdr = Some((ph.vaddr, ph.memsz)),
                PT_TLS if ph.memsz > 0 => {
                    let (s, n) = (ph.offset as usize, ph.filesz as usize);
                    if let Some(init) = s.checked_add(n).and_then(|e| file.get(s..e)) {
                        tls = Some(Tls {
                            vaddr: ph.vaddr,
                            memsz: ph.memsz,
                            align: ph.align.max(1),
                            init,
                        });
                    }
                }
                _ => {}
            }
        }
        // Stable: a file whose segments are not in address order is read in
        // address order.
        segments.sort_by_key(|s| s.vaddr);

        // --- section headers -------------------------------------------------
        let mut shdrs: Vec<Shdr> = Vec::new();
        if opts.sections && shoff != 0 && shnum != 0 {
            for i in 0..shnum {
                let at = (shoff as usize).saturating_add(i * shentsize);
                let sh = if fmt.is_64 {
                    (|| {
                        Some(Shdr {
                            name: le32(file, at)?,
                            kind: le32(file, at + 4)?,
                            flags: le64(file, at + 8)?,
                            addr: le64(file, at + 16)?,
                            offset: le64(file, at + 24)?,
                            size: le64(file, at + 32)?,
                            link: le32(file, at + 40)?,
                        })
                    })()
                } else {
                    (|| {
                        Some(Shdr {
                            name: le32(file, at)?,
                            kind: le32(file, at + 4)?,
                            flags: u64::from(le32(file, at + 8)?),
                            addr: u64::from(le32(file, at + 12)?),
                            offset: u64::from(le32(file, at + 16)?),
                            size: u64::from(le32(file, at + 20)?),
                            link: le32(file, at + 24)?,
                        })
                    })()
                };
                match sh {
                    Some(sh) => shdrs.push(sh),
                    None => {
                        shdrs.clear();
                        break;
                    }
                }
            }
        }
        let sec_bytes = |s: &Shdr| -> &'a [u8] {
            let start = s.offset as usize;
            start
                .checked_add(s.size as usize)
                .and_then(|e| file.get(start..e))
                .unwrap_or(&[])
        };
        let shstr: &'a [u8] = shdrs
            .get(shstrndx)
            .filter(|_| shstrndx != 0)
            .map(&sec_bytes)
            .unwrap_or(&[]);
        let have_names = !shstr.is_empty();
        let sections: Vec<Section<'a>> = if have_names {
            shdrs
                .iter()
                .filter(|s| s.flags & SHF_ALLOC != 0)
                .map(|s| Section {
                    name: cstr(shstr, s.name as usize),
                    addr: s.addr,
                    size: s.size,
                    flags: s.flags,
                    nobits: s.kind == SHT_NOBITS,
                    bytes: if s.kind == SHT_NOBITS {
                        &[]
                    } else {
                        sec_bytes(s)
                    },
                })
                .collect()
        } else {
            Vec::new()
        };
        if eh_hdr.is_none()
            && let Some(s) = sections
                .iter()
                .find(|s| s.name == ".eh_frame_hdr" && s.addr != 0 && !s.nobits)
        {
            eh_hdr = Some((s.addr, s.size));
        }

        let mut img = Image {
            file,
            fmt,
            kind,
            machine,
            entry,
            segments,
            sections,
            relocs: Vec::new(),
            symbols: Vec::new(),
            imports: Vec::new(),
            dynamic: Dynamic::default(),
            tls,
            relro,
            phdrs,
            dynsym_count: 0,
            version_needs: Vec::new(),
            versym: None,
            eh_hdr,
            import_names: HashMap::new(),
            reloc_index: OnceLock::new(),
            eh: OnceLock::new(),
        };

        // --- the dynamic table ----------------------------------------------
        let tags: Vec<(i64, u64)> = dynamic_at
            .map(|(va, sz)| img.dynamic_tags(va, sz))
            .unwrap_or_default();
        let tag = |t: i64| tags.iter().find(|(k, _)| *k == t).map(|(_, v)| *v);

        // --- symbols ---------------------------------------------------------
        let dynsym_sec = shdrs.iter().find(|s| s.kind == SHT_DYNSYM);
        let mut symbols = Vec::new();
        if let Some(sec) = dynsym_sec.filter(|_| have_names) {
            let strs = shdrs.get(sec.link as usize).map(&sec_bytes).unwrap_or(&[]);
            img.read_symbols(sec_bytes(sec), strs, Table::Dynsym, &mut symbols);
        } else if let (Some(symtab), Some(strtab)) = (tag(DT_SYMTAB), tag(DT_STRTAB)) {
            let strsz = tag(DT_STRSZ).unwrap_or(0);
            let strs = img
                .read(strtab, strsz as usize)
                .or_else(|| img.bytes_from(strtab))
                .unwrap_or(&[]);
            let n = img.dynsym_entries(&tags);
            let b = img.bytes_from(symtab).unwrap_or(&[]);
            img.read_symbols(
                &b[..b.len().min(n * fmt.sym_size())],
                strs,
                Table::Dynsym,
                &mut symbols,
            );
        }
        img.dynsym_count = symbols.len();
        if let Some(sec) = shdrs
            .iter()
            .find(|s| s.kind == SHT_SYMTAB)
            .filter(|_| have_names)
        {
            let strs = shdrs.get(sec.link as usize).map(&sec_bytes).unwrap_or(&[]);
            img.read_symbols(sec_bytes(sec), strs, Table::Symtab, &mut symbols);
        }
        img.symbols = symbols;

        // --- the dynamic tags ------------------------------------------------
        let strtab = tag(DT_STRTAB);
        let mut dynamic = Dynamic::default();
        for &(t, v) in &tags {
            if t == DT_NEEDED
                && let Some(st) = strtab
                && let Some(b) = img.bytes_from(st.wrapping_add(v))
            {
                let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
                dynamic
                    .needed
                    .push(std::str::from_utf8(&b[..end]).unwrap_or(""));
            }
        }
        dynamic.init = tag(DT_INIT).filter(|&v| v != 0);
        dynamic.fini = tag(DT_FINI).filter(|&v| v != 0);
        dynamic.pltgot = tag(DT_PLTGOT);
        dynamic.table = dynamic_at;
        dynamic.textrel =
            tag(DT_TEXTREL).is_some() || tag(DT_FLAGS).is_some_and(|f| f & DF_TEXTREL != 0);
        let array = |at: i64, size: i64, sec: &str| -> Option<(u64, u64)> {
            match (tag(at), tag(size)) {
                (Some(a), Some(n)) => Some((a, n)),
                _ => img
                    .sections
                    .iter()
                    .find(|s| s.name == sec && !s.nobits && s.size > 0)
                    .map(|s| (s.addr, s.size)),
            }
        };
        dynamic.preinit_array = array(DT_PREINIT_ARRAY, DT_PREINIT_ARRAYSZ, ".preinit_array");
        dynamic.init_array = array(DT_INIT_ARRAY, DT_INIT_ARRAYSZ, ".init_array");
        dynamic.fini_array = array(DT_FINI_ARRAY, DT_FINI_ARRAYSZ, ".fini_array");
        img.dynamic = dynamic;

        // --- versions --------------------------------------------------------
        img.versym = tag(DT_VERSYM);
        if let (Some(mut at), Some(n), Some(st)) = (tag(DT_VERNEED), tag(DT_VERNEEDNUM), strtab) {
            let mut needs = Vec::new();
            for _ in 0..n.min(1024) {
                let Some(b) = img.read(at, 16) else { break };
                let (cnt, file_off, aux, next) = (
                    u64::from(le16(b, 2).unwrap_or(0)),
                    u64::from(le32(b, 4).unwrap_or(0)),
                    u64::from(le32(b, 8).unwrap_or(0)),
                    u64::from(le32(b, 12).unwrap_or(0)),
                );
                let soname = img
                    .bytes_from(st.wrapping_add(file_off))
                    .map(|s| cstr(s, 0))
                    .unwrap_or("");
                let mut a = at.wrapping_add(aux);
                for _ in 0..cnt.min(1024) {
                    let Some(x) = img.read(a, 16) else { break };
                    needs.push((le16(x, 6).unwrap_or(0), soname));
                    let nx = u64::from(le32(x, 12).unwrap_or(0));
                    if nx == 0 {
                        break;
                    }
                    a = a.wrapping_add(nx);
                }
                if next == 0 {
                    break;
                }
                at = at.wrapping_add(next);
            }
            img.version_needs = needs;
        }

        // --- relocations -----------------------------------------------------
        let mut ranges: Vec<(u64, u64, bool)> = Vec::new();
        let mut plt_range = None;
        if let (Some(a), Some(n)) = (tag(DT_RELA), tag(DT_RELASZ)) {
            ranges.push((a, n, true));
        }
        if let (Some(a), Some(n)) = (tag(DT_JMPREL), tag(DT_PLTRELSZ)) {
            // `DT_PLTREL` says RELA or REL; the file's class decides which in
            // every layout this crate reads (x86-64 RELA, i386 REL).
            plt_range = Some(ranges.len());
            ranges.push((a, n, fmt.is_64));
        }
        if let (Some(a), Some(n)) = (tag(DT_REL), tag(DT_RELSZ)) {
            ranges.push((a, n, false));
        }
        let mut relocs = Vec::new();
        let mut plt_relocs = 0..0;
        for (k, &(a, n, rela)) in ranges.iter().enumerate() {
            let before = relocs.len();
            if let Some(b) = img.read(a, n as usize) {
                img.read_relocs(b, rela, &mut relocs);
            }
            if Some(k) == plt_range {
                plt_relocs = before..relocs.len();
            }
        }
        // Allocated relocation sections no tag covers (a static executable's
        // `.rela.plt` of IRELATIVE entries).
        for s in shdrs.iter().filter(|s| {
            matches!(s.kind, SHT_RELA | SHT_REL) && s.flags & SHF_ALLOC != 0 && s.addr != 0
        }) {
            let covered = ranges
                .iter()
                .any(|&(a, n, _)| s.addr >= a && s.addr + s.size <= a + n);
            if !covered {
                img.read_relocs(sec_bytes(s), s.kind == SHT_RELA, &mut relocs);
            }
        }
        img.relocs = relocs;

        img.build_imports(plt_relocs);
        Ok(img)
    }

    // --- construction helpers ------------------------------------------------

    fn dynamic_tags(&self, va: u64, size: u64) -> Vec<(i64, u64)> {
        let mut out = Vec::new();
        let Some(b) = self.read(va, size as usize) else {
            return out;
        };
        let w = self.fmt.word();
        for e in b.chunks_exact(self.fmt.dyn_size()) {
            let (tag, val) = if self.fmt.is_64 {
                (le64(e, 0).unwrap_or(0) as i64, le64(e, 8).unwrap_or(0))
            } else {
                (
                    i64::from(le32(e, 0).unwrap_or(0) as i32),
                    u64::from(le32(e, w).unwrap_or(0)),
                )
            };
            if tag == 0 {
                break;
            }
            out.push((tag, val));
        }
        out
    }

    /// How many entries `DT_SYMTAB` has: from `DT_GNU_HASH` (the highest
    /// chained index), else `DT_HASH`'s `nchain`, else what fits before
    /// `DT_STRTAB`.
    fn dynsym_entries(&self, tags: &[(i64, u64)]) -> usize {
        let tag = |t: i64| tags.iter().find(|(k, _)| *k == t).map(|(_, v)| *v);
        let symtab = tag(DT_SYMTAB).unwrap_or(0);
        if let Some(gh) = tag(DT_GNU_HASH)
            && let Some(b) = self.bytes_from(gh)
            && let (Some(nb), Some(symoff), Some(bloom)) = (le32(b, 0), le32(b, 4), le32(b, 8))
        {
            let bloom_word = if self.fmt.is_64 { 8 } else { 4 };
            let buckets = 16 + bloom as usize * bloom_word;
            let chains = buckets + nb as usize * 4;
            let mut max = 0u32;
            for i in 0..nb as usize {
                let Some(mut idx) = le32(b, buckets + 4 * i) else {
                    break;
                };
                if idx == 0 {
                    continue;
                }
                loop {
                    max = max.max(idx);
                    let Some(c) = idx
                        .checked_sub(symoff)
                        .and_then(|k| le32(b, chains + 4 * k as usize))
                    else {
                        break;
                    };
                    if c & 1 != 0 {
                        break;
                    }
                    idx += 1;
                }
            }
            if max > 0 {
                return max as usize + 1;
            }
            return symoff as usize;
        }
        if let Some(h) = tag(DT_HASH)
            && let Some(b) = self.bytes_from(h)
            && let Some(nchain) = le32(b, 4)
        {
            return nchain as usize;
        }
        let ent = tag(DT_SYMENT)
            .filter(|&e| e > 0)
            .unwrap_or(self.fmt.sym_size() as u64);
        match tag(DT_STRTAB) {
            Some(st) if st > symtab => ((st - symtab) / ent) as usize,
            _ => 0,
        }
    }

    fn read_symbols(
        &self,
        table: &'a [u8],
        strs: &'a [u8],
        which: Table,
        out: &mut Vec<Symbol<'a>>,
    ) {
        let fmt = self.fmt;
        for e in table.chunks_exact(fmt.sym_size()) {
            let (name, info, shndx, value, size) = if fmt.is_64 {
                (
                    le32(e, 0).unwrap_or(0),
                    e[4],
                    le16(e, 6).unwrap_or(0),
                    le64(e, 8).unwrap_or(0),
                    le64(e, 16).unwrap_or(0),
                )
            } else {
                (
                    le32(e, 0).unwrap_or(0),
                    e[12],
                    le16(e, 14).unwrap_or(0),
                    u64::from(le32(e, 4).unwrap_or(0)),
                    u64::from(le32(e, 8).unwrap_or(0)),
                )
            };
            let kind = match info & 0xf {
                2 | 10 => SymbolKind::Func,
                1 => SymbolKind::Object,
                6 => SymbolKind::Tls,
                _ => SymbolKind::Other,
            };
            out.push(Symbol {
                name: cstr(strs, name as usize),
                addr: value,
                size,
                kind,
                defined: shndx != SHN_UNDEF,
                table: which,
            });
        }
    }

    fn read_relocs(&self, b: &[u8], rela: bool, out: &mut Vec<Reloc>) {
        let fmt = self.fmt;
        let size = if rela {
            fmt.rela_size()
        } else {
            fmt.rel_size()
        };
        for e in b.chunks_exact(size) {
            let offset = fmt.uword(e, 0).unwrap_or(0);
            let info = fmt.uword(e, fmt.word()).unwrap_or(0);
            let (ty, sym) = if fmt.is_64 {
                ((info & 0xffff_ffff) as u32, (info >> 32) as u32)
            } else {
                ((info & 0xff) as u32, (info >> 8) as u32)
            };
            let addend = if !rela {
                0
            } else if fmt.is_64 {
                le64(e, 16).unwrap_or(0) as i64
            } else {
                i64::from(le32(e, 8).unwrap_or(0) as i32)
            };
            out.push(Reloc {
                offset,
                kind: self.reloc_kind(ty),
                sym,
                addend,
            });
        }
    }

    fn reloc_kind(&self, ty: u32) -> RelocKind {
        match (self.machine, ty) {
            (EM_X86_64, 1) | (EM_386, 1) => RelocKind::Abs,
            (EM_X86_64 | EM_386, 5) => RelocKind::Copy,
            (EM_X86_64 | EM_386, 6) => RelocKind::GlobDat,
            (EM_X86_64 | EM_386, 7) => RelocKind::JumpSlot,
            (EM_X86_64 | EM_386, 8) => RelocKind::Relative,
            (EM_X86_64, 18) | (EM_386, 14) => RelocKind::TpOff,
            (EM_X86_64, 37) | (EM_386, 42) => RelocKind::IRelative,
            _ => RelocKind::Other(ty),
        }
    }

    /// The imports: every GOT slot a `JUMP_SLOT` / `GLOB_DAT` relocation
    /// fills for a named symbol, the PLT stub that jumps through it, and the
    /// canonical PLT entry of an address-taken import.
    ///
    /// A stub is found by scanning executable memory for `jmp *slot(%rip)`
    /// (`ff 25`, optionally preceded by `bnd` and `endbr64`): a 16-byte
    /// aligned one whose slot has a `JUMP_SLOT` (`.plt`, `.plt.sec`), or an
    /// 8-byte aligned one, nop-padded, whose slot has a `GLOB_DAT`
    /// (`.plt.got`, the stub of an import that is also address-taken).
    fn build_imports(&mut self, plt_relocs: std::ops::Range<usize>) {
        if !matches!(self.machine, EM_X86_64 | EM_386) {
            return;
        }
        let dynsyms = &self.symbols[..self.dynsym_count];
        let mut jump_slots: HashMap<u64, usize> = HashMap::new();
        let mut glob_dat: HashMap<u64, usize> = HashMap::new();
        let mut got_slots: Vec<(u64, usize)> = Vec::new();
        let plt = &self.relocs[plt_relocs.clone()];
        let rest = self.relocs[..plt_relocs.start]
            .iter()
            .chain(&self.relocs[plt_relocs.end..]);
        for r in plt.iter().chain(rest) {
            if !matches!(r.kind, RelocKind::JumpSlot | RelocKind::GlobDat) {
                continue;
            }
            let Some(sym) = dynsyms.get(r.sym as usize) else {
                continue;
            };
            if sym.name.is_empty() {
                continue;
            }
            if r.kind == RelocKind::JumpSlot {
                jump_slots.insert(r.offset, r.sym as usize);
            } else {
                glob_dat.insert(r.offset, r.sym as usize);
            }
            got_slots.push((r.offset, r.sym as usize));
        }
        let mut stubs: Vec<(u64, u64, usize, StubKind)> = Vec::new(); // (stub, slot, dynsym, kind)
        for seg in self
            .segments
            .iter()
            .filter(|s| s.executable() && self.machine == EM_X86_64)
        {
            let b = seg.bytes;
            let mut i = 0usize;
            while i + 6 <= b.len() {
                if b[i] == 0xff && b[i + 1] == 0x25 {
                    let disp = i32::from_le_bytes(b[i + 2..i + 6].try_into().unwrap()) as i64;
                    let slot = (seg.vaddr + i as u64 + 6).wrapping_add(disp as u64);
                    let mut start = i;
                    if start >= 1 && b[start - 1] == 0xf2 {
                        start -= 1;
                    }
                    if start >= 4 && b[start - 4..start] == [0xf3, 0x0f, 0x1e, 0xfa] {
                        start -= 4;
                    }
                    let va = seg.vaddr + start as u64;
                    if let Some(&sym) = jump_slots.get(&slot) {
                        if va & 15 == 0 {
                            stubs.push((va, slot, sym, StubKind::Plt));
                        }
                    } else if let Some(&sym) = glob_dat.get(&slot)
                        && va & 7 == 0
                        && matches!(b.get(i + 6..i + 8), Some([0x66, 0x90]) | Some([0x0f, 0x1f]))
                    {
                        stubs.push((va, slot, sym, StubKind::PltGot));
                    }
                }
                i += 1;
            }
        }
        let mut stubs_of: HashMap<u64, Vec<(u64, StubKind)>> = HashMap::new();
        for &(va, slot, _, kind) in &stubs {
            stubs_of.entry(slot).or_default().push((va, kind));
        }

        // The name of each address, in the order of authority: stubs, GOT
        // slots, then canonical PLT entries.
        let mut names: HashMap<u64, &'a str> = HashMap::new();
        for &(va, _, sym, _) in &stubs {
            names.entry(va).or_insert(self.symbols[sym].name);
        }
        // One import per stub: several stubs may jump through one slot (a
        // runtime's per-library thunks), and each is a name for it.
        let mut imports = Vec::new();
        for &(slot, sym) in &got_slots {
            let s = &self.symbols[sym];
            names.entry(slot).or_insert(s.name);
            let library = self.symbol_library(sym);
            match stubs_of.get(&slot) {
                Some(found) => {
                    for &(va, kind) in found {
                        imports.push(Import {
                            slot: Some(slot),
                            stub: Some(va),
                            stub_kind: Some(kind),
                            name: s.name,
                            library,
                        });
                    }
                }
                None => imports.push(Import {
                    slot: Some(slot),
                    stub: None,
                    stub_kind: None,
                    name: s.name,
                    library,
                }),
            }
        }
        for (idx, s) in self.symbols[..self.dynsym_count].iter().enumerate() {
            if !s.defined && s.addr != 0 && !s.name.is_empty() && !names.contains_key(&s.addr) {
                names.insert(s.addr, s.name);
                imports.push(Import {
                    slot: None,
                    stub: Some(s.addr),
                    stub_kind: None,
                    name: s.name,
                    library: self.symbol_library(idx),
                });
            }
        }
        self.imports = imports;
        self.import_names = names;
    }

    // --- queries --------------------------------------------------------------

    /// The whole input.
    pub fn file(&self) -> &'a [u8] {
        self.file
    }

    pub fn is_64(&self) -> bool {
        self.fmt.is_64
    }

    /// `ET_DYN`: the loader picks the base.
    pub fn is_pie(&self) -> bool {
        self.kind == ElfKind::SharedObject
    }

    pub fn section(&self, name: &str) -> Option<&Section<'a>> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// The `.dynsym` entries (index 0, the null symbol, included).
    pub fn dynsyms(&self) -> &[Symbol<'a>] {
        &self.symbols[..self.dynsym_count]
    }

    /// The `.symtab` entries; empty when stripped or when the section headers
    /// were not read.
    pub fn symtab(&self) -> &[Symbol<'a>] {
        &self.symbols[self.dynsym_count..]
    }

    /// The symbol a relocation names.
    pub fn reloc_symbol(&self, r: &Reloc) -> Option<&Symbol<'a>> {
        self.dynsyms().get(r.sym as usize)
    }

    /// The soname the version requirement of `.dynsym` entry `idx` names.
    pub fn symbol_library(&self, idx: usize) -> Option<&'a str> {
        let at = self.versym?.checked_add(2 * idx as u64)?;
        let v = le16(self.read(at, 2)?, 0)? & 0x7fff;
        self.version_needs
            .iter()
            .find(|(other, _)| *other == v)
            .map(|(_, s)| *s)
            .filter(|s| !s.is_empty())
    }

    /// The imported symbol a PLT stub, a GOT slot or a canonical PLT entry
    /// stands for.
    pub fn import_name(&self, va: u64) -> Option<&'a str> {
        self.import_names.get(&va).copied()
    }

    /// Every address [`import_name`](Self::import_name) answers for.
    pub fn import_names(&self) -> impl Iterator<Item = (u64, &'a str)> + '_ {
        self.import_names.iter().map(|(a, n)| (*a, *n))
    }

    /// The bytes at `va`, if one segment's file bytes hold all `n`.
    pub fn read(&self, va: u64, n: usize) -> Option<&'a [u8]> {
        for s in &self.segments {
            if va >= s.vaddr && va - s.vaddr <= s.bytes.len() as u64 {
                let off = (va - s.vaddr) as usize;
                if let Some(b) = off.checked_add(n).and_then(|e| s.bytes.get(off..e)) {
                    return Some(b);
                }
            }
        }
        None
    }

    /// The bytes from `va` to the end of its segment's file contents.
    pub fn bytes_from(&self, va: u64) -> Option<&'a [u8]> {
        for s in &self.segments {
            if va >= s.vaddr && va - s.vaddr < s.bytes.len() as u64 {
                return Some(&s.bytes[(va - s.vaddr) as usize..]);
            }
        }
        None
    }

    /// The relocation at `va`, if the loader writes one there (the first, when
    /// several do).
    pub fn reloc_at(&self, va: u64) -> Option<&Reloc> {
        let index = self.reloc_index.get_or_init(|| {
            let mut m = HashMap::with_capacity(self.relocs.len());
            for (i, r) in self.relocs.iter().enumerate() {
                m.entry(r.offset).or_insert(i);
            }
            m
        });
        index.get(&va).map(|&i| &self.relocs[i])
    }

    /// The link-time value of the pointer-sized word at `va` as the loader
    /// leaves it, where that is known without the libraries: a `RELATIVE` (or
    /// symbol-less `Abs`) relocation's addend, a PIE's own symbol plus addend,
    /// else the file bytes (a PIE's relocated words may be 0 in the file).
    pub fn word(&self, va: u64) -> Option<u64> {
        if let Some(r) = self.reloc_at(va) {
            let sym = self.reloc_symbol(r).filter(|s| !s.name.is_empty());
            match r.kind {
                RelocKind::Relative => return Some(r.addend as u64),
                RelocKind::Abs if sym.is_none() => return Some(r.addend as u64),
                RelocKind::GlobDat | RelocKind::JumpSlot | RelocKind::Abs => {
                    if let Some(s) = self.reloc_symbol(r)
                        && self.is_pie()
                        && r.sym != 0
                        && s.defined
                        && s.kind != SymbolKind::Tls
                    {
                        return Some(s.addr.wrapping_add(r.addend as u64));
                    }
                }
                _ => {}
            }
        }
        self.read(va, self.fmt.word())
            .and_then(|b| self.fmt.uword(b, 0))
    }

    /// The words of a constructor / destructor array, as [`word`](Self::word)
    /// reads them: `None` when the array is not in the file.
    pub fn array_words(&self, (addr, size): (u64, u64)) -> Option<Vec<u64>> {
        let w = self.fmt.word() as u64;
        self.read(addr, size as usize)?;
        Some(
            (0..size / w)
                .filter_map(|k| self.word(addr + w * k))
                .collect(),
        )
    }

    /// The ELF header with the program header table, as a segment maps it
    /// (when a `PT_LOAD` maps the first bytes of the file), and the
    /// [`notes`](Self::notes), as `[lo, hi)` ranges in address order. They are
    /// neither code nor program data, though the segment that holds them may
    /// be executable (a linker that does not separate code puts them in front
    /// of `.text`).
    pub fn headers(&self) -> Vec<(u64, u64)> {
        let fmt = self.fmt;
        let (phoff, phnum, phentsize) = if fmt.is_64 {
            (
                le64(self.file, 32),
                le16(self.file, 56),
                le16(self.file, 54),
            )
        } else {
            (
                le32(self.file, 28).map(u64::from),
                le16(self.file, 44),
                le16(self.file, 42),
            )
        };
        let mut out = Vec::new();
        if let (Some(phoff), Some(phnum), Some(phentsize)) = (phoff, phnum, phentsize)
            && let Some(first) = self
                .phdrs
                .iter()
                .find(|ph| ph.kind == PT_LOAD && ph.offset == 0)
        {
            let end = phoff + u64::from(phnum) * u64::from(phentsize);
            if end <= first.filesz {
                out.push((first.vaddr, first.vaddr + end));
            }
        }
        out.extend(self.notes());
        out.sort();
        out
    }

    /// The `PT_NOTE` ranges in address order; notes that abut are one range.
    pub fn notes(&self) -> Vec<(u64, u64)> {
        let mut notes: Vec<(u64, u64)> = self
            .phdrs
            .iter()
            .filter(|ph| ph.kind == PT_NOTE && ph.memsz > 0)
            .map(|ph| (ph.vaddr, ph.vaddr + ph.memsz))
            .collect();
        notes.sort();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in notes {
            match merged.last_mut() {
                Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        merged
    }

    /// The loader's own metadata, as merged `[lo, hi)` ranges: the ELF header,
    /// the program headers, `PT_INTERP`, `PT_NOTE`, and the dynamic tables
    /// (symbols, strings, versions, hashes, relocations). No program object
    /// lives there (GNU ld's first segment holds nothing else; lld's goes on
    /// with `.rodata`), so a literal there is a number, not an address.
    pub fn metadata(&self) -> Vec<(u64, u64)> {
        let mut all = self.loader_tables();
        all.extend(self.headers());
        all.extend(
            self.phdrs
                .iter()
                .filter(|ph| ph.kind == PT_PHDR)
                .map(|ph| (ph.vaddr, ph.vaddr + ph.memsz)),
        );
        all.retain(|(a, e)| e > a);
        all.sort();
        // Merged, across the alignment padding between them.
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (a, e) in all {
            match out.last_mut() {
                Some(last) if a <= last.1 + 64 => last.1 = last.1.max(e),
                _ => out.push((a, e)),
            }
        }
        out
    }

    /// Each of the tables the loader reads, other than the file's own headers
    /// ([`headers`](Self::headers)), as its own `[lo, hi)` range in address
    /// order: `PT_INTERP`, and the string, symbol, version, hash and relocation
    /// tables the dynamic table names. Where one ends the next kind of bytes
    /// begins, so their ends are where a reader that walks a table stops.
    pub fn loader_tables(&self) -> Vec<(u64, u64)> {
        let fmt = self.fmt;
        let mut ends: Vec<(u64, u64)> = Vec::new();
        for ph in self.phdrs.iter().filter(|ph| ph.kind == PT_INTERP) {
            ends.push((ph.vaddr, ph.vaddr + ph.memsz));
        }
        let tags = self
            .phdrs
            .iter()
            .find(|p| p.kind == PT_DYNAMIC)
            .map(|p| self.dynamic_tags(p.vaddr, p.memsz))
            .unwrap_or_default();
        if !tags.is_empty() {
            let tag = |t: i64| tags.iter().find(|x| x.0 == t).map(|x| x.1);
            for (at, size) in [
                (DT_STRTAB, DT_STRSZ),
                (DT_RELA, DT_RELASZ),
                (DT_REL, DT_RELSZ),
                (DT_JMPREL, DT_PLTRELSZ),
                (DT_RELR, DT_RELRSZ),
            ] {
                if let (Some(a), Some(n)) = (tag(at), tag(size)) {
                    ends.push((a, a + n));
                }
            }
            // The symbol table runs up to the string table in both layouts;
            // `.gnu.version` has one half-word per symbol.
            let (symtab, strtab, syment) = (
                tag(DT_SYMTAB),
                tag(DT_STRTAB),
                tag(DT_SYMENT).unwrap_or(fmt.sym_size() as u64).max(1),
            );
            if let (Some(s), Some(t)) = (symtab, strtab) {
                ends.push((s, s.max(t)));
                if let Some(v) = tag(DT_VERSYM) {
                    ends.push((v, v + 2 * (t.saturating_sub(s) / syment)));
                }
            }
            // `.gnu.version_r`: a chain of Verneed records, each with its
            // Vernaux chain.
            if let (Some(mut at), Some(n)) = (tag(DT_VERNEED), tag(DT_VERNEEDNUM)) {
                let start = at;
                let mut end = at;
                for _ in 0..n.min(1024) {
                    let Some(b) = self.read(at, 16) else { break };
                    let (cnt, aux, next) = (
                        u64::from(le16(b, 2).unwrap_or(0)),
                        u64::from(le32(b, 8).unwrap_or(0)),
                        u64::from(le32(b, 12).unwrap_or(0)),
                    );
                    end = end.max(at + 16);
                    let mut a = at + aux;
                    for _ in 0..cnt.min(1024) {
                        let Some(x) = self.read(a, 16) else { break };
                        end = end.max(a + 16);
                        let nx = u64::from(le32(x, 12).unwrap_or(0));
                        if nx == 0 {
                            break;
                        }
                        a += nx;
                    }
                    if next == 0 {
                        break;
                    }
                    at += next;
                }
                ends.push((start, end));
            }
            // The hash tables: sized from their headers.
            let nsyms = match (tag(DT_SYMTAB), tag(DT_STRTAB)) {
                (Some(s), Some(t)) if t > s => {
                    (t - s) / tag(DT_SYMENT).unwrap_or(fmt.sym_size() as u64).max(1)
                }
                _ => 0,
            };
            if let Some(a) = tag(DT_HASH)
                && let Some(h) = self.read(a, 8)
            {
                let (nb, nc) = (
                    u64::from(le32(h, 0).unwrap_or(0)),
                    u64::from(le32(h, 4).unwrap_or(0)),
                );
                ends.push((a, a + 4 * (2 + nb + nc)));
            }
            if let Some(a) = tag(DT_GNU_HASH)
                && let Some(h) = self.read(a, 16)
            {
                let w = |i: usize| u64::from(le32(h, 4 * i).unwrap_or(0));
                let (nb, symoff, bloom) = (w(0), w(1), w(2));
                ends.push((
                    a,
                    a + 16 + 8 * bloom + 4 * nb + 4 * nsyms.saturating_sub(symoff),
                ));
            }
        }
        ends.retain(|(a, e)| e > a);
        ends.sort();
        ends
    }

    // --- exception tables ---------------------------------------------------------

    /// The exception tables: every FDE with an LSDA, its call sites, landing
    /// pads and action chains, and the personality from the FDE's CIE. Read on
    /// first use and kept; loading never touches `.eh_frame`.
    pub fn eh(&self) -> &Eh {
        self.eh.get_or_init(|| crate::eh::parse(self))
    }

    /// `.eh_frame_hdr`'s `(address, size)`: `PT_GNU_EH_FRAME`, else the
    /// section of that name. `None` when the file has none, or it was emptied
    /// (`objcopy --remove-section=.eh_frame_hdr` zeroes the segment).
    pub fn eh_frame_hdr(&self) -> Option<(u64, u64)> {
        self.eh_hdr
    }

    /// The `(start, end)` of the unwind tables the image maps: `.eh_frame_hdr`
    /// ([`Image::eh_frame_hdr`]), and `.eh_frame` and `.gcc_except_table` when
    /// the file has section headers (nothing else locates them). Like
    /// [`Image::loader_tables`], they are not program data: a reader looking for
    /// code pointers in data skips them (their pc-relative entries, read as
    /// words, look like addresses by chance).
    pub fn unwind_tables(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = self.eh_hdr.map(|(a, n)| (a, a + n)).into_iter().collect();
        for name in [".eh_frame", ".gcc_except_table"] {
            if let Some(s) = self.section(name).filter(|s| s.size > 0) {
                out.push((s.addr, s.addr + s.size));
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Function starts the file's metadata names, apart from its symbols:
    /// the `.eh_frame` FDE starts (nameless, but they survive `strip`) and
    /// the constructor / destructor entry points the loader reaches through
    /// `DT_INIT` / `DT_FINI` and the `.init_array` family (the crt functions
    /// `frame_dummy` and `__do_global_dtors_aux` have no FDE).
    ///
    /// Only addresses in executable memory are returned, and no FDE start
    /// inside a PLT stub table (its one FDE covers the whole table). Unsorted,
    /// possibly with duplicates. Reads `.eh_frame` (through the header's
    /// search table when there is one).
    pub fn seeds(&self) -> Vec<Seed> {
        let executable = |a: u64| {
            self.segments
                .iter()
                .any(|s| s.executable() && s.contains(a))
        };
        let plt: Vec<(u64, u64)> = self
            .sections
            .iter()
            .filter(|s| matches!(s.name, ".plt" | ".plt.sec" | ".plt.got"))
            .map(|s| (s.addr, s.addr + s.size))
            .collect();
        let mut out = Vec::new();
        for addr in self.eh_function_starts() {
            if !plt.iter().any(|&(s, e)| addr >= s && addr < e) && executable(addr) {
                out.push(Seed {
                    addr,
                    source: SeedSource::EhFrame,
                });
            }
        }
        let d = &self.dynamic;
        let mut init: Vec<u64> = d.init.into_iter().chain(d.fini).collect();
        for arr in [d.preinit_array, d.init_array, d.fini_array]
            .into_iter()
            .flatten()
        {
            init.extend(self.array_words(arr).unwrap_or_default());
        }
        for addr in init {
            // `0` and `-1` are the linker's empty / sentinel slots.
            if addr != 0 && addr != u64::MAX && executable(addr) {
                out.push(Seed {
                    addr,
                    source: SeedSource::InitFini,
                });
            }
        }
        out
    }

    /// The FDE starts: `.eh_frame_hdr`'s search table, else a walk over
    /// `.eh_frame`. Unsorted, possibly with duplicates.
    pub fn eh_function_starts(&self) -> Vec<u64> {
        let mem = |va: u64| self.bytes_from(va);
        let ptr = self.fmt.word();
        if let Some((hdr, len)) = self.eh_hdr
            && let Some(starts) = crate::eh_frame::hdr_starts(&mem, hdr, len, ptr)
        {
            return starts;
        }
        match self
            .sections
            .iter()
            .find(|s| s.name == ".eh_frame" && s.addr != 0 && !s.nobits)
        {
            Some(s) => crate::eh_frame::stream_starts(&mem, s.addr, ptr),
            None => Vec::new(),
        }
    }
}
