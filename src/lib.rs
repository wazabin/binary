//! Binary container formats and the [`BinaryFormat`] trait.
//!
//! This is a leaf crate below `qcode`: it holds the [`Arch`] enum, the
//! [`TargetOs`] enum, and the ELF/PE/blob container parsers behind a single
//! [`BinaryFormat`] trait. `qcode` re-exports [`TargetOs`] and (in Stage 2b)
//! implements [`BinaryFormat`] for its serializable `MemoryImage`; `harbinger`
//! re-exports this crate as `harbinger::format` and `harbinger::arch::Arch`.

mod arch;
mod target_os;

pub mod blob;
pub mod elf;
pub mod pe;

pub use arch::Arch;
pub use target_os::TargetOs;

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

    /// Return the symbol name for the function starting at `addr`, if the
    /// binary format has one (e.g. from an ELF symbol table).
    /// Returns `None` for formats with no symbol information.
    fn symbol_name(&self, _addr: u64) -> Option<&str> {
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
    /// This is **not** the negation of [`is_known_writable`]: a format that
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
    use super::{BinaryFormat, blob::Blob};

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
