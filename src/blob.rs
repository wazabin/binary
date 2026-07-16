use crate::Arch;

use crate::BinaryFormat;

/// A generic binary blob: raw bytes loaded at a fixed virtual address.
///
/// This is the simplest possible [`BinaryFormat`] — it holds an arbitrary byte
/// slice together with its load address and has no symbol information.  Tests
/// and other code that already work with raw bytes can use this instead of
/// passing `(address, bytes)` tuples directly, enabling them to go through the
/// same `BinaryFormat`-based pipeline as real ELF binaries.
#[derive(Clone)]
pub struct Blob {
    pub load_address: u64,
    pub data: Vec<u8>,
}

impl Blob {
    pub fn new(load_address: u64, data: Vec<u8>) -> Self {
        Self { load_address, data }
    }

    /// Borrow `bytes` directly without copying.
    pub fn from_slice(load_address: u64, bytes: &[u8]) -> Self {
        Self {
            load_address,
            data: bytes.to_vec(),
        }
    }
}

impl BinaryFormat for Blob {
    fn load_address(&self) -> u64 {
        self.load_address
    }

    fn architecture(&self) -> Arch {
        // X64 by default
        Arch::X86_64
    }

    /// The only entry point is the load address itself.
    fn entry_points(&self) -> Vec<u64> {
        vec![self.load_address]
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        let offset = addr.checked_sub(self.load_address)? as usize;
        self.data.get(offset).copied()
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        let offset = addr.checked_sub(self.load_address)? as usize;
        self.data.get(offset..)
    }

    fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.contains(addr).then(|| {
            (
                self.load_address,
                self.load_address + self.data.len() as u64,
            )
        })
    }

    /// A blob is one region; treat it as executable (raw code/data) so resolved
    /// jump targets in it are not filtered out.
    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        vec![(self.load_address, self.data.clone(), true, false)]
    }
}
