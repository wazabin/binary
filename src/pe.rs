use crate::Arch;
use goblin::pe::{
    PE,
    header::{
        COFF_MACHINE_ARM, COFF_MACHINE_ARM64, COFF_MACHINE_ARMNT, COFF_MACHINE_X86,
        COFF_MACHINE_X86_64,
    },
    symbol::Symbol as CoffSymbol,
};

use crate::symbols::SymbolIndex;
use crate::{BinaryFormat, Endian, Section, Symbol};

/// A known function start address extracted from PE metadata.
#[derive(Debug, Clone)]
pub struct PeFunctionSymbol {
    /// Virtual address of the function.
    pub address: u64,
    /// Symbol name, if present.
    pub name: Option<String>,
    /// The function's size from its exception-directory (`RUNTIME_FUNCTION`)
    /// entry. `None` when only a COFF symbol names it, or for import thunks.
    pub size: Option<u64>,
    /// Whether this is an external imported function.
    pub is_external: bool,
    /// For an import thunk, the DLL the import comes from.
    pub library: Option<String>,
}

/// A PE import resolved from the import address table.
#[derive(Debug, Clone)]
pub struct PeImportSymbol {
    /// Virtual address of the import address table slot.
    pub iat_address: u64,
    /// Bare imported function name used for display.
    pub name: String,
    /// Source DLL name.
    pub dll: String,
    /// Import ordinal, if present.
    pub ordinal: Option<u16>,
}

/// Results of PE analysis.
#[derive(Debug, Clone)]
pub struct PeAnalysis {
    /// Preferred image-base virtual address.
    pub image_base: u64,
    /// PE optional-header entrypoint VA.
    pub entrypoint: u64,
    /// Function start addresses discovered from COFF symbols and imports.
    pub known_functions: Vec<PeFunctionSymbol>,
    /// Imported functions keyed by their IAT slots.
    pub imports: Vec<PeImportSymbol>,
}

#[derive(Debug, Clone)]
pub struct PeSection {
    /// The section-table name, e.g. `.text`; empty when the header has none.
    pub name: String,
    pub start: u64,
    pub mem_size: u64,
    pub data: Vec<u8>,
    /// `IMAGE_SCN_MEM_WRITE` from the section characteristics. Read straight
    /// from the section table, so a section without the bit is proven
    /// read-only.
    pub writable: bool,
    /// `IMAGE_SCN_MEM_EXECUTE` from the section characteristics. Reported by
    /// [`BinaryFormat::sections`]; the loader's [`BinaryFormat::mapped_regions`]
    /// deliberately stays permissive and marks every section executable.
    pub executable: bool,
}

/// `IMAGE_SCN_MEM_EXECUTE`: the section may be executed as code.
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
/// `IMAGE_SCN_MEM_WRITE`: the section is writable at run time.
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

impl PeSection {
    fn end(&self) -> u64 {
        self.start + self.mem_size
    }

    fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end()
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        if !self.contains(addr) {
            return None;
        }
        let offset = (addr - self.start) as usize;
        Some(self.data.get(offset).copied().unwrap_or(0))
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        if !self.contains(addr) {
            return None;
        }
        let offset = (addr - self.start) as usize;
        self.data.get(offset..)
    }
}

/// A parsed and loaded PE32/PE32+ executable image.
///
/// `analysis.known_functions` is sorted by address with one entry per
/// address, and a name index is built alongside it, which is what the
/// [`BinaryFormat::symbol_name`] / [`BinaryFormat::symbol_address`] lookups
/// search. Both hold once parsing returns; mutating the table afterwards is
/// not supported.
#[derive(Debug, Clone)]
pub struct PeBinary {
    pub load_address: u64,
    pub sections: Vec<PeSection>,
    pub analysis: PeAnalysis,
    /// Function names -> addresses, over `analysis.known_functions`.
    symbols: SymbolIndex,
    pub architecture: Arch,
    pub is_64: bool,
}

/// Errors that can occur while parsing a PE binary.
#[derive(Debug)]
pub enum PeError {
    Parse(goblin::error::Error),
    NoMappedSection,
    UnsupportedMachine(u16),
}

impl std::fmt::Display for PeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeError::Parse(e) => write!(f, "PE parse error: {e}"),
            PeError::NoMappedSection => write!(f, "PE contains no mapped sections"),
            PeError::UnsupportedMachine(machine) => {
                write!(f, "PE has unsupported machine type 0x{machine:x}")
            }
        }
    }
}

impl std::error::Error for PeError {}

impl From<goblin::error::Error> for PeError {
    fn from(e: goblin::error::Error) -> Self {
        PeError::Parse(e)
    }
}

impl PeBinary {
    /// Parse a PE32/PE32+ executable from raw file bytes.
    pub fn parse(file_bytes: &[u8]) -> Result<Self, PeError> {
        let pe = PE::parse(file_bytes)?;
        let architecture = from_pe_machine(pe.header.coff_header.machine)
            .ok_or(PeError::UnsupportedMachine(pe.header.coff_header.machine))?;

        let mut sections = pe
            .sections
            .iter()
            .filter(|s| s.virtual_size > 0 || s.size_of_raw_data > 0)
            .map(|section| {
                let start = pe.image_base + u64::from(section.virtual_address);
                let mem_size = u64::from(section.virtual_size.max(section.size_of_raw_data));
                let offset = section.pointer_to_raw_data as usize;
                let filesz = section.size_of_raw_data as usize;
                let data = file_bytes
                    .get(offset..offset.saturating_add(filesz))
                    .unwrap_or(&[])
                    .to_vec();
                PeSection {
                    name: section
                        .real_name
                        .clone()
                        .or_else(|| section.name().ok().map(str::to_owned))
                        .unwrap_or_default(),
                    start,
                    mem_size,
                    data,
                    writable: section.characteristics & IMAGE_SCN_MEM_WRITE != 0,
                    executable: section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
                }
            })
            .filter(|s| s.mem_size > 0)
            .collect::<Vec<_>>();

        if sections.is_empty() {
            return Err(PeError::NoMappedSection);
        }
        sections.sort_by_key(|s| s.start);
        let load_address = sections[0].start;

        let entrypoint = pe.image_base + u64::from(pe.entry);
        let mut known_functions = Vec::new();
        collect_coff_function_symbols(&pe, file_bytes, &mut known_functions);
        let imports: Vec<_> = pe
            .imports
            .iter()
            .map(|import| {
                let name = import.name.to_string();
                PeImportSymbol {
                    iat_address: pe.image_base + import.offset as u64,
                    name,
                    dll: import.dll.to_string(),
                    ordinal: (import.ordinal != 0).then_some(import.ordinal),
                }
            })
            .collect();
        mark_import_thunks(&mut known_functions, &sections, &imports, pe.is_64);
        collect_exception_function_symbols(&pe, &sections, &mut known_functions);
        uniquify_function_names(&mut known_functions);

        // The sorted order is what `function_at` binary-searches. A COFF
        // symbol and an exception-directory entry for the same function
        // collapse into the symbol's entry, which inherits the entry's size.
        known_functions.sort_by_key(|f| (f.address, !f.is_external, f.name.is_none()));
        known_functions.dedup_by(|dropped, kept| {
            if dropped.address != kept.address {
                return false;
            }
            if kept.size.is_none() {
                kept.size = dropped.size;
            }
            true
        });
        let symbols = SymbolIndex::build(
            known_functions
                .iter()
                .map(|f| (f.name.as_deref(), f.address, f.is_external)),
        );

        Ok(Self {
            load_address,
            sections,
            analysis: PeAnalysis {
                image_base: pe.image_base,
                entrypoint,
                known_functions,
                imports,
            },
            symbols,
            architecture,
            is_64: pe.is_64,
        })
    }

    /// The known function starting at `addr`, by binary search over the
    /// address-sorted table.
    fn function_at(&self, addr: u64) -> Option<&PeFunctionSymbol> {
        let functions = &self.analysis.known_functions;
        let idx = functions.binary_search_by_key(&addr, |f| f.address).ok()?;
        Some(&functions[idx])
    }

    fn import_at_iat(&self, addr: u64) -> Option<&PeImportSymbol> {
        self.analysis
            .imports
            .iter()
            .find(|import| import.iat_address == addr)
    }
}

fn collect_exception_function_symbols(
    pe: &PE<'_>,
    sections: &[PeSection],
    out: &mut Vec<PeFunctionSymbol>,
) {
    let Some(exception_data) = pe.exception_data.as_ref() else {
        return;
    };

    for function in exception_data.functions().flatten() {
        if function.begin_address == 0 || function.end_address <= function.begin_address {
            continue;
        }

        let address = pe.image_base + u64::from(function.begin_address);
        if !sections.iter().any(|section| section.contains(address)) {
            continue;
        }

        out.push(PeFunctionSymbol {
            address,
            name: None,
            size: Some(u64::from(function.end_address - function.begin_address)),
            is_external: false,
            library: None,
        });
    }
}

fn collect_coff_function_symbols(pe: &PE<'_>, file_bytes: &[u8], out: &mut Vec<PeFunctionSymbol>) {
    let Ok(Some(symbols)) = pe.header.coff_header.symbols(file_bytes) else {
        return;
    };
    let strings = pe.header.coff_header.strings(file_bytes).ok().flatten();

    for (_index, inline_name, symbol) in symbols.iter() {
        if !symbol.is_function_definition() {
            continue;
        }
        let section_index = (symbol.section_number - 1) as usize;
        let Some(section) = pe.sections.get(section_index) else {
            continue;
        };
        let Some(name) = symbol_name(&symbol, inline_name, strings.as_ref()) else {
            continue;
        };
        let address = pe.image_base + u64::from(section.virtual_address) + u64::from(symbol.value);
        out.push(PeFunctionSymbol {
            address,
            name: Some(name),
            size: None,
            is_external: false,
            library: None,
        });
    }
}

fn symbol_name(
    symbol: &CoffSymbol,
    inline_name: Option<&str>,
    strings: Option<&goblin::strtab::Strtab<'_>>,
) -> Option<String> {
    if let Some(name) = inline_name
        && !name.is_empty()
    {
        return Some(normalize_coff_name(name));
    }

    if let Some(strings) = strings
        && let Ok(name) = symbol.name(strings)
        && !name.is_empty()
    {
        return Some(normalize_coff_name(name));
    }

    let end = symbol
        .name
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(symbol.name.len());
    let name = std::str::from_utf8(&symbol.name[..end]).ok()?;
    (!name.is_empty()).then(|| normalize_coff_name(name))
}

fn normalize_coff_name(name: &str) -> String {
    name.strip_prefix('_').unwrap_or(name).to_string()
}

fn mark_import_thunks(
    functions: &mut [PeFunctionSymbol],
    sections: &[PeSection],
    imports: &[PeImportSymbol],
    is_64: bool,
) {
    for function in functions {
        let Some(iat_address) = import_thunk_iat(sections, function.address, is_64) else {
            continue;
        };
        let Some(import) = imports
            .iter()
            .find(|import| import.iat_address == iat_address)
        else {
            continue;
        };

        function.name = Some(import.name.clone());
        function.is_external = true;
        function.library = Some(import.dll.clone());
    }
}

fn import_thunk_iat(sections: &[PeSection], address: u64, is_64: bool) -> Option<u64> {
    let bytes = sections
        .iter()
        .find(|section| section.contains(address))
        .and_then(|section| section.bytes_at(address))?;

    if bytes.len() < 6 || bytes[0] != 0xff || bytes[1] != 0x25 {
        return None;
    }

    let operand = [bytes[2], bytes[3], bytes[4], bytes[5]];
    if is_64 {
        let disp = i32::from_le_bytes(operand) as i64;
        Some((address as i64 + 6 + disp) as u64)
    } else {
        Some(u32::from_le_bytes(operand) as u64)
    }
}

fn uniquify_function_names(functions: &mut [PeFunctionSymbol]) {
    functions.sort_by_key(|function| {
        (
            function.name.clone(),
            !function.is_external,
            function.address,
        )
    });

    let mut names = std::collections::HashSet::new();
    for function in functions {
        let Some(name) = function.name.as_mut() else {
            continue;
        };
        if names.insert(name.clone()) {
            continue;
        }

        let base = name.clone();
        let mut candidate = format!("{base}_{:x}", function.address);
        let mut suffix = 1;
        while !names.insert(candidate.clone()) {
            candidate = format!("{base}_{:x}_{suffix}", function.address);
            suffix += 1;
        }
        *name = candidate;
    }
}

impl BinaryFormat for PeBinary {
    fn load_address(&self) -> u64 {
        self.load_address
    }

    fn architecture(&self) -> Arch {
        self.architecture
    }

    fn os(&self) -> crate::TargetOs {
        crate::TargetOs::Windows
    }

    /// PE is little-endian on every machine this crate maps.
    fn endianness(&self) -> Endian {
        Endian::Little
    }

    fn bits(&self) -> u32 {
        if self.is_64 { 64 } else { 32 }
    }

    /// The section table, with the real `IMAGE_SCN_MEM_EXECUTE` bit (unlike
    /// [`mapped_regions`](BinaryFormat::mapped_regions), which stays
    /// permissive).
    fn sections(&self) -> Vec<Section> {
        self.sections
            .iter()
            .map(|s| Section {
                name: s.name.clone(),
                address: s.start,
                size: s.mem_size,
                writable: s.writable,
                executable: s.executable,
            })
            .collect()
    }

    fn symbols(&self) -> Vec<Symbol> {
        self.analysis
            .known_functions
            .iter()
            .filter_map(|f| {
                let name = f.name.as_deref().filter(|n| !n.is_empty())?;
                Some(Symbol {
                    name: name.to_owned(),
                    address: f.address,
                    size: f.size,
                    is_external: f.is_external,
                    library: f.library.clone(),
                })
            })
            .collect()
    }

    /// Import-directory DLL names in first-seen order, deduplicated
    /// case-insensitively but reported in their original case.
    fn linked_libraries(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for import in &self.analysis.imports {
            if !seen.iter().any(|dll| dll.eq_ignore_ascii_case(&import.dll)) {
                seen.push(import.dll.clone());
            }
        }
        seen
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        self.sections
            .iter()
            .find(|section| section.contains(addr))
            .and_then(|section| section.byte_at(addr))
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        self.sections
            .iter()
            .find(|section| section.contains(addr))
            .and_then(|section| section.bytes_at(addr))
    }

    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        self.sections
            .iter()
            .map(|sec| {
                let mut bytes = sec.data.clone();
                bytes.resize(sec.mem_size as usize, 0);
                // Mark every mapped section executable so resolved jump targets
                // are not filtered out (the flag is only a permissive sanity
                // check). Writability is the real `IMAGE_SCN_MEM_WRITE` bit, so
                // a snapshot built from these regions can tell `.rdata` from
                // `.data` exactly as the live binary does.
                (sec.start, bytes, true, sec.writable)
            })
            .collect()
    }

    /// A mapped section without `IMAGE_SCN_MEM_WRITE` is proven read-only.
    /// (There is deliberately no `is_known_writable` override: flipping that
    /// predicate would change which constants the folding passes trust, which is
    /// a separate question from proving a region immutable.)
    fn is_known_read_only(&self, addr: u64) -> bool {
        self.sections
            .iter()
            .any(|section| !section.writable && section.contains(addr))
    }

    fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.sections
            .iter()
            .find(|section| section.contains(addr))
            .map(|section| (section.start, section.end()))
    }

    fn symbol_name(&self, addr: u64) -> Option<&str> {
        self.function_at(addr)
            .and_then(|f| f.name.as_deref())
            .filter(|name| !name.is_empty())
            .or_else(|| (addr == self.analysis.entrypoint).then_some("_start"))
    }

    /// The inverse of [`symbol_name`](BinaryFormat::symbol_name), including
    /// the synthetic `_start` it gives an unnamed entrypoint.
    fn symbol_address(&self, name: &str) -> Option<u64> {
        self.symbols.address(name).or_else(|| {
            let entry = self.analysis.entrypoint;
            (name == "_start" && self.symbol_name(entry) == Some("_start")).then_some(entry)
        })
    }

    fn is_external_symbol(&self, addr: u64) -> bool {
        self.function_at(addr).is_some_and(|f| f.is_external)
    }

    fn entry_points(&self) -> Vec<u64> {
        let mut entries = vec![self.analysis.entrypoint];
        let has_named_code_symbols = self
            .analysis
            .known_functions
            .iter()
            .any(|f| !f.is_external && f.name.is_some());

        for f in &self.analysis.known_functions {
            if f.is_external {
                continue;
            }

            if (!has_named_code_symbols && f.name.is_none())
                || f.name.as_deref().is_some_and(is_primary_pe_function_name)
            {
                entries.push(f.address);
            }
        }
        entries.sort();
        entries.dedup();
        entries
    }

    fn entrypoint(&self) -> Option<u64> {
        Some(self.analysis.entrypoint)
    }

    fn import_symbol_name(&self, addr: u64) -> Option<&str> {
        self.import_at_iat(addr).map(|import| import.name.as_str())
    }

    fn import_library(&self, addr: u64) -> Option<&str> {
        // Externals are minted either at an import thunk's address or (for
        // direct `call [iat]` sites with no thunk) at the IAT slot itself.
        self.import_at_iat(addr)
            .map(|import| import.dll.as_str())
            .or_else(|| self.function_at(addr).and_then(|f| f.library.as_deref()))
    }
}

fn is_primary_pe_function_name(name: &str) -> bool {
    matches!(name, "main" | "WinMain" | "wmain" | "wWinMain")
}

pub fn from_pe_machine(value: u16) -> Option<Arch> {
    match value {
        COFF_MACHINE_X86 => Some(Arch::I386),
        COFF_MACHINE_X86_64 => Some(Arch::X86_64),
        COFF_MACHINE_ARM | COFF_MACHINE_ARMNT => Some(Arch::Arm),
        COFF_MACHINE_ARM64 => Some(Arch::AArch64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section-less image around `analysis`, indexed the way `parse` does it.
    fn binary(mut analysis: PeAnalysis) -> PeBinary {
        analysis
            .known_functions
            .sort_by_key(|f| (f.address, !f.is_external, f.name.is_none()));
        analysis.known_functions.dedup_by_key(|f| f.address);
        let symbols = SymbolIndex::build(
            analysis
                .known_functions
                .iter()
                .map(|f| (f.name.as_deref(), f.address, f.is_external)),
        );
        PeBinary {
            load_address: 0x400000,
            sections: vec![],
            analysis,
            symbols,
            architecture: Arch::I386,
            is_64: false,
        }
    }

    fn function(address: u64, name: &str, library: Option<&str>) -> PeFunctionSymbol {
        PeFunctionSymbol {
            address,
            name: Some(name.to_string()),
            size: None,
            is_external: library.is_some(),
            library: library.map(str::to_string),
        }
    }

    #[test]
    fn import_symbol_name_returns_iat_import_name() {
        let pe = binary(PeAnalysis {
            image_base: 0x400000,
            entrypoint: 0x401000,
            known_functions: vec![],
            imports: vec![PeImportSymbol {
                iat_address: 0x404000,
                name: "ExitProcess".to_string(),
                dll: "KERNEL32.DLL".to_string(),
                ordinal: None,
            }],
        });

        assert_eq!(pe.import_symbol_name(0x404000), Some("ExitProcess"));
        assert_eq!(pe.import_symbol_name(0x404004), None);
    }

    #[test]
    fn entrypoint_returns_primary_pe_entrypoint() {
        let pe = binary(PeAnalysis {
            image_base: 0x400000,
            entrypoint: 0x401000,
            known_functions: vec![function(0x402000, "main", None)],
            imports: vec![],
        });

        assert_eq!(pe.entrypoint(), Some(0x401000));
        assert!(pe.entry_points().contains(&0x401000));
    }

    #[test]
    fn symbol_lookup_by_name_and_address_are_inverses() {
        let pe = binary(PeAnalysis {
            image_base: 0x400000,
            entrypoint: 0x401000,
            known_functions: vec![
                function(0x402000, "main", None),
                function(0x401100, "helper", None),
                function(0x403000, "ExitProcess", Some("KERNEL32.DLL")),
            ],
            imports: vec![],
        });

        assert_eq!(pe.symbol_address("main"), Some(0x402000));
        assert_eq!(pe.symbol_address("helper"), Some(0x401100));
        assert_eq!(pe.symbol_address("ExitProcess"), Some(0x403000));
        assert_eq!(pe.symbol_address("missing"), None);
        assert_eq!(pe.symbol_name(0x402000), Some("main"));
        assert_eq!(pe.symbol_name(0x401100), Some("helper"));
        assert_eq!(pe.symbol_name(0x403000), Some("ExitProcess"));
        assert_eq!(pe.symbol_name(0x402001), None);
        assert!(pe.is_external_symbol(0x403000));
        assert!(!pe.is_external_symbol(0x402000));
        assert_eq!(pe.import_library(0x403000), Some("KERNEL32.DLL"));

        // The unnamed entrypoint reads as `_start` both ways.
        assert_eq!(pe.symbol_name(0x401000), Some("_start"));
        assert_eq!(pe.symbol_address("_start"), Some(0x401000));
    }

    #[test]
    fn enumerates_sections_and_named_symbols() {
        let mut pe = binary(PeAnalysis {
            image_base: 0x400000,
            entrypoint: 0x401000,
            known_functions: vec![
                function(0x402000, "main", None),
                PeFunctionSymbol {
                    address: 0x402100,
                    name: None,
                    size: Some(0x40),
                    is_external: false,
                    library: None,
                },
                function(0x403000, "ExitProcess", Some("KERNEL32.DLL")),
            ],
            imports: vec![],
        });
        pe.sections = vec![
            PeSection {
                name: ".text".to_string(),
                start: 0x401000,
                mem_size: 0x1000,
                data: vec![],
                writable: false,
                executable: true,
            },
            PeSection {
                name: ".data".to_string(),
                start: 0x404000,
                mem_size: 0x200,
                data: vec![],
                writable: true,
                executable: false,
            },
        ];

        assert_eq!(
            pe.sections(),
            vec![
                Section {
                    name: ".text".to_string(),
                    address: 0x401000,
                    size: 0x1000,
                    writable: false,
                    executable: true,
                },
                Section {
                    name: ".data".to_string(),
                    address: 0x404000,
                    size: 0x200,
                    writable: true,
                    executable: false,
                },
            ]
        );
        // The nameless exception-directory entry is a function start, not
        // a symbol.
        assert_eq!(
            pe.symbols(),
            vec![
                Symbol {
                    name: "main".to_string(),
                    address: 0x402000,
                    size: None,
                    is_external: false,
                    library: None,
                },
                Symbol {
                    name: "ExitProcess".to_string(),
                    address: 0x403000,
                    size: None,
                    is_external: true,
                    library: Some("KERNEL32.DLL".to_string()),
                },
            ]
        );
        assert_eq!(pe.bits(), 32);
        assert_eq!(pe.endianness(), Endian::Little);
    }

    #[test]
    fn named_entrypoint_is_not_also_start() {
        let pe = binary(PeAnalysis {
            image_base: 0x400000,
            entrypoint: 0x401000,
            known_functions: vec![function(0x401000, "mainCRTStartup", None)],
            imports: vec![],
        });

        assert_eq!(pe.symbol_name(0x401000), Some("mainCRTStartup"));
        assert_eq!(pe.symbol_address("mainCRTStartup"), Some(0x401000));
        assert_eq!(pe.symbol_address("_start"), None);
    }
}
