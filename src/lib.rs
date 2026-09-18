//! Binary container formats and the [`BinaryFormat`] trait.
//!
//! This is a leaf crate below `qcode`: it holds the [`Arch`] enum, the
//! [`TargetOs`] enum, and the ELF/PE/blob container parsers behind a single
//! [`BinaryFormat`] trait. `qcode` re-exports [`TargetOs`] and (in Stage 2b)
//! implements [`BinaryFormat`] for its serializable `MemoryImage`; `harbinger`
//! re-exports this crate as `harbinger::format` and `harbinger::arch::Arch`.

mod arch;
mod symbols;
mod target_os;

pub mod blob;
pub mod eh_frame;
pub mod elf;
pub mod pe;

pub use arch::Arch;
pub use target_os::TargetOs;

/// The byte order a container was parsed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

/// A container format this crate can parse, as sniffed from a file's magic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `\x7fELF`.
    Elf,
    /// An `MZ` stub whose `e_lfanew` points at a `PE\0\0` signature.
    Pe,
}

impl Format {
    /// Sniff the container format from the start of a file, or `None` when
    /// the bytes match neither (a raw blob, or a bare DOS `MZ` executable).
    pub fn detect(bytes: &[u8]) -> Option<Format> {
        if bytes.starts_with(b"\x7fELF") {
            return Some(Format::Elf);
        }
        if bytes.starts_with(b"MZ") {
            let e_lfanew = bytes.get(0x3c..0x40)?;
            let at = u32::from_le_bytes(e_lfanew.try_into().ok()?) as usize;
            if bytes.get(at..at + 4) == Some(b"PE\0\0") {
                return Some(Format::Pe);
            }
        }
        None
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Format::Elf => "elf",
            Format::Pe => "pe",
        })
    }
}

/// Why [`load`] could not produce a binary.
#[derive(Debug)]
pub enum LoadError {
    /// The bytes are neither ELF nor PE; callers that accept raw images can
    /// fall back to [`blob::Blob`] with a load address of their choosing.
    UnknownFormat,
    Elf(elf::ElfError),
    Pe(pe::PeError),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::UnknownFormat => write!(f, "not an ELF or PE file"),
            LoadError::Elf(e) => write!(f, "failed to parse ELF: {e}"),
            LoadError::Pe(e) => write!(f, "failed to parse PE: {e}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::UnknownFormat => None,
            LoadError::Elf(e) => Some(e),
            LoadError::Pe(e) => Some(e),
        }
    }
}

/// Detect the container format of `bytes` and parse it.
///
/// This is the entry point for callers that do not care which container
/// they were handed: the result answers every [`BinaryFormat`] query. Match
/// on [`Format::detect`] and call [`elf::ElfBinary::parse`] or
/// [`pe::PeBinary::parse`] directly to keep the concrete type.
pub fn load(bytes: &[u8]) -> Result<Box<dyn BinaryFormat>, LoadError> {
    match Format::detect(bytes).ok_or(LoadError::UnknownFormat)? {
        Format::Elf => elf::ElfBinary::parse(bytes)
            .map(|b| Box::new(b) as Box<dyn BinaryFormat>)
            .map_err(LoadError::Elf),
        Format::Pe => pe::PeBinary::parse(bytes)
            .map(|b| Box::new(b) as Box<dyn BinaryFormat>)
            .map_err(LoadError::Pe),
    }
}

/// A named region of the image, as the container's own layout table lists
/// it: an ELF section header or a PE section. Returned by
/// [`BinaryFormat::sections`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// The name from the container's string table, e.g. `.text`.
    pub name: String,
    /// Link-time virtual address of the first byte.
    pub address: u64,
    /// Size in memory, zero-filled tail (`.bss`) included.
    pub size: u64,
    /// The container's writable flag (`SHF_WRITE`, `IMAGE_SCN_MEM_WRITE`).
    pub writable: bool,
    /// The container's executable flag (`SHF_EXECINSTR`,
    /// `IMAGE_SCN_MEM_EXECUTE`).
    pub executable: bool,
}

/// A named function symbol, as returned by [`BinaryFormat::symbols`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    /// Link-time virtual address of the function's first byte.
    pub address: u64,
    /// The function's size in bytes, when the container records one (ELF
    /// `st_size`, a PE exception-directory entry). `None` when it does not,
    /// or records zero.
    pub size: Option<u64>,
    /// Whether this is an import stub (PLT entry, IAT thunk) rather than a
    /// function defined in the image.
    pub is_external: bool,
    /// For an import stub, the library it resolves to, when known.
    pub library: Option<String>,
}

/// Map a byte to its printable ASCII representation.
///
/// Returns `b` unchanged for bytes in the printable ASCII range (0x20–0x7e),
/// and `b'.'` for everything else. Used by hex viewers for the ASCII column.
pub const fn printable_byte(b: u8) -> u8 {
    if b >= 0x20 && b <= 0x7e { b } else { b'.' }
}

/// Return `true` when `b` is a printable ASCII byte.
///
/// This intentionally excludes ASCII control characters such as newline and
/// tab. It is meant for filtering C strings while solving binaries, where
/// accepting control bytes tends to produce noisy false positives.
pub const fn is_printable_ascii_byte(b: u8) -> bool {
    b >= 0x20 && b <= 0x7e
}

/// A trait representing a loaded binary image with one or more mapped regions
/// and one or more known entry points.
///
/// Both [`blob::Blob`] (raw bytes) and [`elf::ElfBinary`] implement this trait
/// so that arch disassemblers can accept either format through a single
/// interface, even when the backing image is sparse.
///
/// Impls are immutable after parse, so the trait carries `Send + Sync` bounds:
/// a parsed binary is shared by reference (`Arc<dyn BinaryFormat>`) across the
/// analysis pipeline's worker threads.
pub trait BinaryFormat: Send + Sync {
    /// The lowest virtual address mapped by this binary image.
    fn load_address(&self) -> u64;

    /// Return the byte mapped at `addr`, or `None` if the address is unmapped.
    fn byte_at(&self, addr: u64) -> Option<u8>;

    /// Return the contiguous bytes available at `addr`.
    ///
    /// The returned slice covers the current mapped region from `addr` onward.
    /// Implementations may return an owned buffer when the region is backed by
    /// implicit zero-fill rather than file bytes.
    fn bytes_at(&self, addr: u64) -> Option<&[u8]>;

    /// Known entry points into this binary (entrypoint, symbol table functions, etc.).
    /// The recursive disassembler should be seeded with these addresses.
    fn entry_points(&self) -> Vec<u64>;

    /// The binary's primary entry point (e.g. the ELF `e_entry`), if the format
    /// designates one. Unlike [`entry_points`], this is the single address where
    /// execution begins. Returns `None` for formats with no distinguished entry.
    ///
    /// [`entry_points`]: Self::entry_points
    fn entrypoint(&self) -> Option<u64> {
        None
    }

    /// Name of the architecture, should use embedded information from known binary
    /// format, or raw values from the blob.
    fn architecture(&self) -> Arch;

    /// The byte order the container declares (ELF `EI_DATA`); PE is always
    /// little-endian. Defaults to little-endian for formats that record none.
    fn endianness(&self) -> Endian {
        Endian::Little
    }

    /// The container's word size in bits, 32 or 64 (ELF `EI_CLASS`, PE32 vs
    /// PE32+). Defaults to 64 for formats that record none.
    fn bits(&self) -> u32 {
        64
    }

    /// The operating system this binary targets, inferred from the container
    /// format. Defaults to [`TargetOs::Unknown`]; PE returns Windows, ELF Linux.
    fn os(&self) -> TargetOs {
        TargetOs::Unknown
    }

    /// Library names this binary links against: ELF `DT_NEEDED` sonames,
    /// PE import-directory DLL names (original case; matching is
    /// case-insensitive). Empty when the format has no such notion (Blob).
    fn linked_libraries(&self) -> Vec<String> {
        Vec::new()
    }

    /// The container's layout table, in address order: an ELF file's
    /// allocated section headers, a PE file's sections. Coarser than
    /// [`mapped_regions`] for ELF (one `PT_LOAD` holds several sections) and
    /// finer-named. Empty for formats with no such table (Blob).
    ///
    /// [`mapped_regions`]: Self::mapped_regions
    fn sections(&self) -> Vec<Section> {
        Vec::new()
    }

    /// Every named function symbol, in address order: what
    /// [`symbol_name`](Self::symbol_name) would answer for each address, with
    /// the size and import provenance the container records. Empty for
    /// formats with no symbol information.
    fn symbols(&self) -> Vec<Symbol> {
        Vec::new()
    }

    /// Return the symbol name for the function starting at `addr`, if the
    /// binary format has one (e.g. from an ELF symbol table).
    /// Returns `None` for formats with no symbol information.
    ///
    /// `addr` is a link-time address, as [`load_address`] and [`entry_points`]
    /// report them: for a position-independent executable or shared object
    /// the loader adds a base of its choosing, so subtract that base from a
    /// run-time address before asking. The ELF and PE implementations answer
    /// from an index built at parse time rather than scanning their function
    /// table, so a caller may ask per address. The inverse lookup is
    /// [`symbol_address`].
    ///
    /// [`load_address`]: Self::load_address
    /// [`entry_points`]: Self::entry_points
    /// [`symbol_address`]: Self::symbol_address
    fn symbol_name(&self, _addr: u64) -> Option<&str> {
        None
    }

    /// Return the address of the function symbol called `name`, if the binary
    /// format has one (e.g. `main` from an ELF `.symtab`, or an import stub
    /// named after the function it resolves to).
    /// Returns `None` for formats with no symbol information.
    ///
    /// The address is the link-time one, as [`load_address`] and
    /// [`entry_points`] report them: for a position-independent executable or
    /// shared object it is relative to whatever base the loader picks, so add
    /// that base to reach the function at run time. When several functions
    /// share a name, a defined function is preferred over an import stub, and
    /// otherwise the lowest address is returned. The ELF and PE
    /// implementations answer from an index built at parse time rather than
    /// scanning their function table. The inverse lookup is [`symbol_name`].
    ///
    /// [`load_address`]: Self::load_address
    /// [`entry_points`]: Self::entry_points
    /// [`symbol_name`]: Self::symbol_name
    fn symbol_address(&self, _name: &str) -> Option<u64> {
        None
    }

    /// Returns `true` if `addr` is an external (imported) function stub,
    /// e.g. a PLT thunk.  The recursive disassembler will not lift the body
    /// of external functions.  Defaults to `false`.
    fn is_external_symbol(&self, _addr: u64) -> bool {
        false
    }

    /// Return the name of the library providing the external function stub at
    /// `addr`: the PE import-directory DLL, or the ELF `.gnu.version_r` soname
    /// the symbol's version requirement points at. `None` when the format does
    /// not record a per-symbol source library (e.g. an unversioned ELF import).
    fn import_library(&self, _addr: u64) -> Option<&str> {
        None
    }

    /// Return the imported symbol whose resolver slot lives at `addr`, if any.
    ///
    /// ELF uses this for GOT / PLT relocation slots such as `R_X86_64_GLOB_DAT`
    /// and `R_X86_64_JUMP_SLOT`.
    fn import_symbol_name(&self, _addr: u64) -> Option<&str> {
        None
    }

    /// Return rows of bytes suitable for a hex viewer.
    ///
    /// The first row is aligned down to the nearest `width`-byte boundary.
    /// Each row is `(row_address, bytes)` where a byte is `None` when the
    /// address is unmapped (e.g. a hole in a sparse ELF image).
    ///
    /// # Panics
    /// Panics if `width` is zero.
    fn hex_rows(&self, addr: u64, len: usize, width: usize) -> Vec<(u64, Vec<Option<u8>>)> {
        assert!(width > 0, "hex row width must be non-zero");
        let row_start = addr - addr % width as u64;
        let end = addr.saturating_add(len as u64);
        let mut rows = Vec::new();
        let mut cur = row_start;
        while cur < end {
            let bytes = (0..width as u64).map(|i| self.byte_at(cur + i)).collect();
            rows.push((cur, bytes));
            cur = cur.saturating_add(width as u64);
        }
        rows
    }

    /// Return `true` if `addr` is mapped by this binary image.
    fn contains(&self, addr: u64) -> bool {
        self.byte_at(addr).is_some()
    }

    /// Enumerate the binary's mapped regions as
    /// `(start, bytes, executable, writable)`.
    ///
    /// At snapshot-save time the persistence layer copies these into a
    /// serializable `MemoryImage` so a reloaded session is self-describing.
    /// Each region's `bytes` should cover its full in-memory size (zero-filled
    /// tail included), mirroring [`byte_at`].
    ///
    /// Defaults to empty for formats that do not expose their segments.
    ///
    /// [`byte_at`]: Self::byte_at
    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        Vec::new()
    }

    /// Return `true` if `addr` lies in an executable region of this binary
    /// image. Defaults to "mapped" for formats that don't track per-region
    /// permissions; formats with permission information (e.g. ELF segment
    /// flags) should override this.
    fn is_executable(&self, addr: u64) -> bool {
        self.contains(addr)
    }

    /// The `[start, end)` bounds of the mapped region containing `addr`, if
    /// any. Used to key per-segment facts (e.g. executability propositions)
    /// so repeated queries in one region collapse to a single entry. Defaults
    /// to `None` for formats that do not expose their segments.
    fn segment_bounds(&self, _addr: u64) -> Option<(u64, u64)> {
        None
    }

    /// Return `true` only if `addr` lies in a region *known* to be writable
    /// (from the container's segment flags). Passes that fold a value out of
    /// initialized memory use this to refuse mutable memory — e.g. a GOT slot
    /// the dynamic linker overwrites at load time. Defaults to `false`
    /// ("not proven writable"); `Blob` keeps the default.
    fn is_known_writable(&self, _addr: u64) -> bool {
        false
    }

    /// Return `true` only if `addr` lies in a region *proven* read-only (mapped,
    /// and the container's own permission data says the region is not writable).
    ///
    /// This is **not** the negation of [`is_known_writable`](Self::is_known_writable): a format that
    /// records no permissions answers `false` to both, which reads as "unknown"
    /// rather than "read-only". Consumers that reconstruct a *value* out of
    /// initialized memory — the decompiler rendering a `.rodata` string constant
    /// as a named object — need the positive proof, because a writable byte may
    /// be a different byte at run time.
    ///
    /// Defaults to `false` ("not proven read-only"); `Blob` and `PeBinary` keep
    /// the default, the latter because it does not parse section permissions.
    fn is_known_read_only(&self, _addr: u64) -> bool {
        false
    }

    /// Read `n` bytes at virtual address `addr`.
    ///
    /// Returns `None` if any byte in the requested range is unmapped.
    fn read_bytes(&self, addr: u64, n: usize) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(n);
        for offset in 0..n {
            out.push(self.byte_at(addr.checked_add(offset as u64)?)?);
        }
        Some(out)
    }

    /// Read a little-endian unsigned integer of `size` bytes (1..=8) at `addr`.
    ///
    /// Returns `None` if the size is out of range or any byte is unmapped.
    fn read_uint(&self, addr: u64, size: usize) -> Option<u64> {
        if size == 0 || size > 8 {
            return None;
        }
        let bytes = self.read_bytes(addr, size)?;
        let mut value = 0u64;
        for (i, &b) in bytes.iter().enumerate() {
            value |= (b as u64) << (i * 8);
        }
        Some(value)
    }

    /// Read a null-terminated C string at virtual address `addr`, returning the
    /// bytes up to (but not including) the null terminator.
    ///
    /// If `max_len` is `Some(limit)`, only the first `limit` bytes are searched
    /// for the null terminator. Returns `None` if the address is out of range or
    /// no null terminator is found within the (optionally limited) region.
    fn read_cstring(&self, addr: u64, max_len: Option<usize>) -> Option<Vec<u8>> {
        let limit = max_len.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        for offset in 0..limit {
            let byte = self.byte_at(addr.checked_add(offset as u64)?)?;
            if byte == 0 {
                return Some(out);
            }
            out.push(byte);
        }
        None
    }

    /// Read a null-terminated C string at virtual address `addr`, requiring
    /// every byte before the null terminator to be printable ASCII.
    ///
    /// Printable bytes are in the inclusive range `0x20..=0x7e`. Returns
    /// `None` if the address is out of range, no null terminator is found
    /// within the optional search window, or a non-printable byte appears
    /// before the terminator.
    fn read_printable_cstring(&self, addr: u64, max_len: Option<usize>) -> Option<Vec<u8>> {
        let limit = max_len.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        for offset in 0..limit {
            let byte = self.byte_at(addr.checked_add(offset as u64)?)?;
            if byte == 0 {
                return Some(out);
            }
            if !is_printable_ascii_byte(byte) {
                return None;
            }
            out.push(byte);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryFormat, Endian, Format, LoadError, blob::Blob, load};

    #[test]
    fn detects_elf_by_magic() {
        assert_eq!(Format::detect(b"\x7fELF\x02\x01\x01"), Some(Format::Elf));
        assert_eq!(Format::detect(b"\x7fEL"), None);
    }

    #[test]
    fn detects_pe_only_behind_a_pe_signature() {
        let mut pe = vec![0u8; 0x80];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c..0x40].copy_from_slice(&0x40u32.to_le_bytes());
        assert_eq!(Format::detect(&pe), None);
        pe[0x40..0x44].copy_from_slice(b"PE\0\0");
        assert_eq!(Format::detect(&pe), Some(Format::Pe));

        // A bare DOS executable, or an `MZ` too short to hold `e_lfanew`.
        assert_eq!(Format::detect(b"MZ\x90\x00"), None);
        let mut dos = vec![0u8; 0x80];
        dos[..2].copy_from_slice(b"MZ");
        assert_eq!(Format::detect(&dos), None);
    }

    #[test]
    fn detects_nothing_in_a_blob() {
        assert_eq!(Format::detect(b""), None);
        assert_eq!(Format::detect(&[0x55, 0x48, 0x89, 0xe5]), None);
        assert!(matches!(load(b"\x90\x90"), Err(LoadError::UnknownFormat)));
    }

    #[test]
    fn blob_keeps_the_trait_defaults() {
        let blob = Blob::new(0x1000, vec![0x90]);
        assert_eq!(blob.endianness(), Endian::Little);
        assert_eq!(blob.bits(), 64);
        assert!(blob.sections().is_empty());
        assert!(blob.symbols().is_empty());
    }

    #[test]
    fn read_printable_cstring_accepts_printable_ascii() {
        let blob = Blob::new(0x1000, b"hello, world!\0next".to_vec());

        assert_eq!(
            blob.read_printable_cstring(0x1000, None),
            Some(b"hello, world!".to_vec())
        );
    }

    #[test]
    fn read_printable_cstring_rejects_control_bytes() {
        let blob = Blob::new(0x1000, b"line\nbreak\0".to_vec());

        assert_eq!(
            blob.read_cstring(0x1000, None),
            Some(b"line\nbreak".to_vec())
        );
        assert_eq!(blob.read_printable_cstring(0x1000, None), None);
    }

    #[test]
    fn read_printable_cstring_rejects_non_ascii_bytes() {
        let blob = Blob::new(0x1000, b"caf\xe9\0".to_vec());

        assert_eq!(blob.read_printable_cstring(0x1000, None), None);
    }

    #[test]
    fn read_printable_cstring_honors_max_len() {
        let blob = Blob::new(0x1000, b"hello\0".to_vec());

        assert_eq!(blob.read_printable_cstring(0x1000, Some(3)), None);
        assert_eq!(
            blob.read_printable_cstring(0x1000, Some(6)),
            Some(b"hello".to_vec())
        );
    }
}
