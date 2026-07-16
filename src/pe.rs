use crate::Arch;
use goblin::pe::{
    PE,
    header::{
        COFF_MACHINE_ARM, COFF_MACHINE_ARM64, COFF_MACHINE_ARMNT, COFF_MACHINE_X86,
        COFF_MACHINE_X86_64,
    },
    symbol::Symbol,
};

use crate::BinaryFormat;

/// A known function start address extracted from PE metadata.
#[derive(Debug, Clone)]
pub struct PeFunctionSymbol {
    /// Virtual address of the function.
    pub address: u64,
    /// Symbol name, if present.
    pub name: Option<String>,
    /// Whether this is an external imported function.
    pub is_external: bool,
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
    pub start: u64,
    pub mem_size: u64,
    pub data: Vec<u8>,
}

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
#[derive(Debug, Clone)]
pub struct PeBinary {
    pub load_address: u64,
    pub sections: Vec<PeSection>,
    pub analysis: PeAnalysis,
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
                    start,
                    mem_size,
                    data,
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

        known_functions.sort_by_key(|f| (f.address, !f.is_external, f.name.is_none()));
        known_functions.dedup_by_key(|f| f.address);

        Ok(Self {
            load_address,
            sections,
            analysis: PeAnalysis {
                image_base: pe.image_base,
                entrypoint,
                known_functions,
                imports,
            },
            architecture,
            is_64: pe.is_64,
        })
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
            is_external: false,
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
            is_external: false,
        });
    }
}

fn symbol_name(
    symbol: &Symbol,
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
                // `PeSection` does not record per-section permissions; mark
                // every mapped section executable so resolved jump targets are
                // not filtered out (the flag is only a permissive sanity check),
                // and non-writable so the constant-read guard keeps today's
                // behavior (PE imports are resolved via the IAT, not here).
                (sec.start, bytes, true, false)
            })
            .collect()
    }

    fn symbol_name(&self, addr: u64) -> Option<&str> {
        self.analysis
            .known_functions
            .iter()
            .find(|f| f.address == addr)
            .and_then(|f| f.name.as_deref())
            .filter(|name| !name.is_empty())
            .or_else(|| (addr == self.analysis.entrypoint).then_some("_start"))
    }

    fn is_external_symbol(&self, addr: u64) -> bool {
        self.analysis
            .known_functions
            .iter()
            .find(|f| f.address == addr)
            .map(|f| f.is_external)
            .unwrap_or(false)
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

    #[test]
    fn import_symbol_name_returns_iat_import_name() {
        let pe = PeBinary {
            load_address: 0x400000,
            sections: vec![],
            analysis: PeAnalysis {
                image_base: 0x400000,
                entrypoint: 0x401000,
                known_functions: vec![],
                imports: vec![PeImportSymbol {
                    iat_address: 0x404000,
                    name: "ExitProcess".to_string(),
                    dll: "KERNEL32.DLL".to_string(),
                    ordinal: None,
                }],
            },
            architecture: Arch::I386,
            is_64: false,
        };

        assert_eq!(pe.import_symbol_name(0x404000), Some("ExitProcess"));
        assert_eq!(pe.import_symbol_name(0x404004), None);
    }

    #[test]
    fn entrypoint_returns_primary_pe_entrypoint() {
        let pe = PeBinary {
            load_address: 0x400000,
            sections: vec![],
            analysis: PeAnalysis {
                image_base: 0x400000,
                entrypoint: 0x401000,
                known_functions: vec![PeFunctionSymbol {
                    address: 0x402000,
                    name: Some("main".to_string()),
                    is_external: false,
                }],
                imports: vec![],
            },
            architecture: Arch::I386,
            is_64: false,
        };

        assert_eq!(pe.entrypoint(), Some(0x401000));
        assert!(pe.entry_points().contains(&0x401000));
    }
}
