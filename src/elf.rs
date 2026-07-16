use elf::{
    ElfBytes,
    abi::{
        DT_NEEDED, EM_386, EM_X86_64, PF_W, PF_X, PT_LOAD, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT,
        SHT_DYNSYM, SHT_REL, SHT_RELA, SHT_SYMTAB, STT_FUNC,
    },
    endian::AnyEndian,
};

use crate::Arch;

use crate::BinaryFormat;

/// A known function start address extracted from an ELF symbol table.
#[derive(Debug, Clone)]
pub struct FunctionSymbol {
    /// Virtual address of the function.
    pub address: u64,
    /// Name from the string table, if present.
    pub name: Option<String>,
    /// Whether this is an external (imported) function stub (e.g. a PLT thunk).
    pub is_external: bool,
}

/// A resolver slot for an imported function.
#[derive(Debug, Clone)]
pub struct ImportSymbol {
    /// Virtual address of the GOT / PLT relocation slot.
    pub address: u64,
    /// Imported function name.
    pub name: String,
}

/// Results of the ELF analysis passes.
#[derive(Debug, Clone)]
pub struct ElfAnalysis {
    /// The binary entry point (`e_entry` from the ELF header).
    pub entrypoint: u64,
    /// Function start addresses discovered via `.symtab` / `.dynsym`.
    pub known_functions: Vec<FunctionSymbol>,
    /// Imported functions keyed by resolver slot address.
    pub imported_symbols: Vec<ImportSymbol>,
}

#[derive(Debug, Clone)]
pub struct LoadSegment {
    pub start: u64,
    pub mem_size: u64,
    pub data: Vec<u8>,
    /// Whether the segment is executable (`PF_X` set in `p_flags`).
    pub executable: bool,
    /// Whether the segment is writable (`PF_W` set in `p_flags`).
    pub writable: bool,
}

impl LoadSegment {
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

/// A parsed and loaded ELF binary.
///
/// Every `PT_LOAD` segment is recorded as a mapped region. File-backed bytes
/// are preserved per segment, and any `p_memsz > p_filesz` tail is modeled as
/// implicit zero-fill rather than materialized up front.
///
/// [`ElfBinary::analysis`] is populated by two analysis passes run at parse
/// time:
/// - **Entrypoint**: `e_entry` from the ELF header.
/// - **Known functions**: all `STT_FUNC` symbols found in `.symtab` or
///   `.dynsym`.
#[derive(Debug, Clone)]
pub struct ElfBinary {
    pub load_address: u64,
    pub segments: Vec<LoadSegment>,
    pub analysis: ElfAnalysis,
    pub architecture: Arch,
    /// Shared-library sonames from the `.dynamic` section's `DT_NEEDED` entries,
    /// in link order (e.g. `libc.so.6`).
    pub needed_libraries: Vec<String>,
}

/// Errors that can occur while parsing an ELF binary.
#[derive(Debug)]
pub enum ElfError {
    /// The underlying `elf` crate returned an error.
    Parse(elf::ParseError),
    /// The ELF file contains no loadable (`PT_LOAD`) segment.
    NoLoadableSegment,
    /// The ELF file header number does not match any arch
    UnknownArch,
}

impl std::fmt::Display for ElfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElfError::Parse(e) => write!(f, "ELF parse error: {e}"),
            ElfError::NoLoadableSegment => write!(f, "ELF contains no PT_LOAD segment"),
            ElfError::UnknownArch => write!(f, "ELF has an unknown arch"),
        }
    }
}

impl std::error::Error for ElfError {}

impl From<elf::ParseError> for ElfError {
    fn from(e: elf::ParseError) -> Self {
        ElfError::Parse(e)
    }
}

impl ElfBinary {
    /// Parse an ELF binary from raw file bytes.
    ///
    /// Records every `PT_LOAD` segment as a mapped region, reads the entrypoint
    /// from the ELF header, and collects all `STT_FUNC` symbols from `.symtab`
    /// and `.dynsym`.
    pub fn parse(file_bytes: &[u8]) -> Result<Self, ElfError> {
        let elf = ElfBytes::<AnyEndian>::minimal_parse(file_bytes)?;

        // --- Load the first PT_LOAD segment ---------------------------------
        let segments = elf.segments().ok_or(ElfError::NoLoadableSegment)?;

        let mut load_segments: Vec<_> = segments
            .iter()
            .filter(|p| p.p_type == PT_LOAD && p.p_memsz > 0)
            .map(|p| {
                let offset = p.p_offset as usize;
                let filesz = p.p_filesz as usize;
                let data = file_bytes
                    .get(offset..offset + filesz)
                    .unwrap_or(&[])
                    .to_vec();
                LoadSegment {
                    start: p.p_vaddr,
                    mem_size: p.p_memsz,
                    data,
                    executable: p.p_flags & PF_X != 0,
                    writable: p.p_flags & PF_W != 0,
                }
            })
            .collect();

        if load_segments.is_empty() {
            return Err(ElfError::NoLoadableSegment);
        }
        load_segments.sort_by_key(|s| s.start);
        let load_address = load_segments[0].start;

        // --- Analysis pass 1: entrypoint ------------------------------------
        let entrypoint = elf.ehdr.e_entry;

        // --- Analysis pass 2: known function starts -------------------------
        let mut known_functions: Vec<FunctionSymbol> = Vec::new();
        collect_func_symbols(&elf, SHT_SYMTAB, &mut known_functions);
        collect_func_symbols(&elf, SHT_DYNSYM, &mut known_functions);

        // --- Analysis pass 3: PLT stub names -----------------------------------
        collect_plt_symbols(&elf, &mut known_functions);

        // --- Analysis pass 4: GOT / PLT resolver slots -------------------------
        let mut imported_symbols = Vec::new();
        collect_import_symbols(&elf, &mut imported_symbols);

        // --- Analysis pass 5: DT_NEEDED shared-library sonames -----------------
        let needed_libraries = collect_needed_libraries(&elf);

        // Deduplicate by address (symtab may overlap dynsym).
        known_functions.sort_by_key(|f| f.address);
        known_functions.dedup_by_key(|f| f.address);
        imported_symbols.sort_by_key(|f| f.address);
        imported_symbols.dedup_by_key(|f| f.address);

        Ok(ElfBinary {
            load_address,
            segments: load_segments,
            analysis: ElfAnalysis {
                entrypoint,
                known_functions,
                imported_symbols,
            },
            architecture: from_elf_machine(elf.ehdr.e_machine).ok_or(ElfError::UnknownArch)?,
            needed_libraries,
        })
    }
}

/// Collect `DT_NEEDED` shared-library sonames from the `.dynamic` section,
/// resolving each entry's string-table offset against `.dynstr`. Returns an
/// empty vector for statically linked binaries (no dynamic section).
fn collect_needed_libraries(elf: &ElfBytes<AnyEndian>) -> Vec<String> {
    let mut out = Vec::new();
    let Some((_symtab, dynstr)) = elf.dynamic_symbol_table().ok().flatten() else {
        return out;
    };
    let Ok(Some(dynamic)) = elf.dynamic() else {
        return out;
    };
    for entry in dynamic.iter() {
        if entry.d_tag == DT_NEEDED
            && let Ok(name) = dynstr.get(entry.d_val() as usize)
            && !name.is_empty()
        {
            out.push(name.to_owned());
        }
    }
    out
}

/// Collect all `STT_FUNC` symbols from the section matching `section_type`
/// (either `SHT_SYMTAB` or `SHT_DYNSYM`) into `out`.
fn collect_func_symbols(
    elf: &ElfBytes<AnyEndian>,
    section_type: u32,
    out: &mut Vec<FunctionSymbol>,
) {
    if section_type == SHT_SYMTAB {
        let Ok(Some((symtab, strtab))) = elf.symbol_table() else {
            return;
        };
        for sym in symtab.iter() {
            if sym.st_symtype() == STT_FUNC && sym.st_value != 0 {
                let name = strtab.get(sym.st_name as usize).ok().map(str::to_owned);
                out.push(FunctionSymbol {
                    address: sym.st_value,
                    name,
                    is_external: false,
                });
            }
        }
    } else if section_type == SHT_DYNSYM {
        let Ok(Some((dynsymtab, dynstrtab))) = elf.dynamic_symbol_table() else {
            return;
        };
        for sym in dynsymtab.iter() {
            if sym.st_symtype() == STT_FUNC && sym.st_value != 0 {
                let name = dynstrtab.get(sym.st_name as usize).ok().map(str::to_owned);
                out.push(FunctionSymbol {
                    address: sym.st_value,
                    name,
                    is_external: false,
                });
            }
        }
    }
}

/// Collect PLT stub addresses and names from `.rel[a].plt` + `.dynsym`.
///
/// Each relocation entry at index `i` (0-based) corresponds to the PLT stub at
/// `plt_start + (i + 1) * entry_size`. The stub is named after the imported
/// symbol, not `symbol@plt`, so direct calls render as calls to the external.
fn collect_plt_symbols(elf: &ElfBytes<AnyEndian>, out: &mut Vec<FunctionSymbol>) {
    let _ = collect_plt_symbols_inner(elf, out);
}

fn collect_plt_symbols_inner(
    elf: &ElfBytes<AnyEndian>,
    out: &mut Vec<FunctionSymbol>,
) -> Option<()> {
    let (shdrs, shstrtab) = match elf.section_headers_with_strtab() {
        Ok((Some(s), Some(st))) => (s, st),
        _ => return None,
    };

    let section_addr = |name: &str| {
        shdrs
            .iter()
            .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(name))
            .map(|s| s.sh_addr)
    };

    let plt_start = section_addr(".plt")?;
    let plt_entry_size = plt_entry_size(elf.ehdr.e_machine)?;
    // CET-enabled binaries emit a second, parallel stub table `.plt.sec`: one
    // 16-byte `endbr64; bnd jmp *GOT` entry per import (no reserved entry 0), in
    // the same order as `.rela.plt`. Direct `call`s go through `.plt.sec`, while
    // `.plt` is reached only by the lazy resolver — so when it exists, the import
    // name belongs on the `.plt.sec` stub the callers reference.
    let plt_sec_start = section_addr(".plt.sec");

    let (dynsymtab, dynstrtab) = elf.dynamic_symbol_table().ok().flatten()?;

    // Import relocations from `.rela.plt` (x86-64 / RELA) or `.rel.plt`
    // (x86-32 / REL), as `(index, symbol_name)` pairs in section order.
    let mut import_relocs: Vec<(usize, Option<String>)> = Vec::new();
    if let Some(rela_plt_shdr) = shdrs
        .iter()
        .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(".rela.plt"))
    {
        for (i, rela) in elf.section_data_as_relas(&rela_plt_shdr).ok()?.enumerate() {
            if is_import_relocation(elf.ehdr.e_machine, rela.r_type) {
                let name = dyn_symbol_name(&dynsymtab, &dynstrtab, rela.r_sym).map(str::to_string);
                import_relocs.push((i, name));
            }
        }
    } else if let Some(rel_plt_shdr) = shdrs
        .iter()
        .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(".rel.plt"))
    {
        for (i, rel) in elf.section_data_as_rels(&rel_plt_shdr).ok()?.enumerate() {
            if is_import_relocation(elf.ehdr.e_machine, rel.r_type) {
                let name = dyn_symbol_name(&dynsymtab, &dynstrtab, rel.r_sym).map(str::to_string);
                import_relocs.push((i, name));
            }
        }
    }

    for (i, name) in import_relocs {
        // `.plt` stub for relocation `i` sits after the reserved entry 0.
        let plt_stub_addr = plt_start + ((i + 1) as u64) * plt_entry_size;
        match plt_sec_start {
            // With a `.plt.sec`, that stub bears the import name (callers target
            // it); the `.plt` stub stays external but nameless, so it is not
            // mistaken for a pure local — and so its name does not collide with
            // the `.plt.sec` stub's.
            Some(sec_start) => {
                out.push(FunctionSymbol {
                    address: sec_start + (i as u64) * plt_entry_size,
                    name,
                    is_external: true,
                });
                out.push(FunctionSymbol {
                    address: plt_stub_addr,
                    name: None,
                    is_external: true,
                });
            }
            None => out.push(FunctionSymbol {
                address: plt_stub_addr,
                name,
                is_external: true,
            }),
        }
    }

    Some(())
}

fn dyn_symbol_name<'a>(
    dynsymtab: &elf::symbol::SymbolTable<'a, AnyEndian>,
    dynstrtab: &elf::string_table::StringTable<'a>,
    sym_idx: u32,
) -> Option<&'a str> {
    let sym = dynsymtab.get(sym_idx as usize).ok()?;
    let name = dynstrtab.get(sym.st_name as usize).ok()?;
    if name.is_empty() { None } else { Some(name) }
}

fn is_import_relocation(machine: u16, r_type: u32) -> bool {
    match machine {
        EM_386 | EM_X86_64 => r_type == R_X86_64_JUMP_SLOT,
        _ => false,
    }
}

fn plt_entry_size(machine: u16) -> Option<u64> {
    match machine {
        EM_386 | EM_X86_64 => Some(16),
        _ => None,
    }
}

fn collect_import_symbols(elf: &ElfBytes<AnyEndian>, out: &mut Vec<ImportSymbol>) {
    let _ = collect_import_symbols_inner(elf, out);
}

fn collect_import_symbols_inner(
    elf: &ElfBytes<AnyEndian>,
    out: &mut Vec<ImportSymbol>,
) -> Option<()> {
    let (shdrs, _) = match elf.section_headers_with_strtab() {
        Ok((Some(s), _)) => (s, ()),
        _ => return None,
    };

    let (dynsymtab, dynstrtab) = elf.dynamic_symbol_table().ok().flatten()?;

    for shdr in shdrs.iter() {
        match shdr.sh_type {
            SHT_RELA => {
                let relas = elf.section_data_as_relas(&shdr).ok()?;
                for rela in relas {
                    if !is_got_import_relocation(rela.r_type) {
                        continue;
                    }
                    let sym = dynsymtab.get(rela.r_sym as usize).ok()?;
                    let name = dynstrtab.get(sym.st_name as usize).ok()?.to_owned();
                    out.push(ImportSymbol {
                        address: rela.r_offset,
                        name,
                    });
                }
            }
            SHT_REL => {
                let rels = elf.section_data_as_rels(&shdr).ok()?;
                for rel in rels {
                    if !is_got_import_relocation(rel.r_type) {
                        continue;
                    }
                    let sym = dynsymtab.get(rel.r_sym as usize).ok()?;
                    let name = dynstrtab.get(sym.st_name as usize).ok()?.to_owned();
                    out.push(ImportSymbol {
                        address: rel.r_offset,
                        name,
                    });
                }
            }
            _ => {}
        }
    }

    Some(())
}

fn is_got_import_relocation(r_type: u32) -> bool {
    matches!(r_type, R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT)
}

impl BinaryFormat for ElfBinary {
    fn load_address(&self) -> u64 {
        self.load_address
    }

    fn architecture(&self) -> Arch {
        self.architecture
    }

    fn os(&self) -> crate::TargetOs {
        crate::TargetOs::Linux
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        self.segments
            .iter()
            .find(|segment| segment.contains(addr))
            .and_then(|segment| segment.byte_at(addr))
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        self.segments
            .iter()
            .find(|segment| segment.contains(addr))
            .and_then(|segment| segment.bytes_at(addr))
    }

    fn is_executable(&self, addr: u64) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.executable && segment.contains(addr))
    }

    fn is_known_writable(&self, addr: u64) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.writable && segment.contains(addr))
    }

    fn linked_libraries(&self) -> Vec<String> {
        self.needed_libraries.clone()
    }

    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        self.segments
            .iter()
            .map(|seg| {
                // Materialize the `p_memsz > p_filesz` zero-fill tail so reads
                // line up with `byte_at` (which returns 0 there).
                let mut bytes = seg.data.clone();
                bytes.resize(seg.mem_size as usize, 0);
                (seg.start, bytes, seg.executable, seg.writable)
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
            .or_else(|| {
                (addr == self.analysis.entrypoint && self.is_executable(addr)).then_some("_start")
            })
    }

    fn is_external_symbol(&self, addr: u64) -> bool {
        self.analysis
            .known_functions
            .iter()
            .find(|f| f.address == addr)
            .map(|f| f.is_external)
            .unwrap_or(false)
    }

    fn import_symbol_name(&self, addr: u64) -> Option<&str> {
        self.analysis
            .imported_symbols
            .iter()
            .find(|f| f.address == addr)
            .map(|f| f.name.as_str())
    }

    /// Entry points are the ELF entrypoint plus all known function starts.
    fn entry_points(&self) -> Vec<u64> {
        let mut entries = vec![self.analysis.entrypoint];
        for f in &self.analysis.known_functions {
            entries.push(f.address);
        }
        entries.sort();
        entries.dedup();
        entries
    }

    /// The ELF entry point (`e_entry` from the header).
    fn entrypoint(&self) -> Option<u64> {
        Some(self.analysis.entrypoint)
    }
}

pub fn from_elf_machine(value: u16) -> Option<Arch> {
    match value {
        1 => Some(Arch::M32),
        2 => Some(Arch::Sparc),
        3 => Some(Arch::I386),
        4 => Some(Arch::M68K),
        5 => Some(Arch::M88K),
        7 => Some(Arch::I860),
        8 => Some(Arch::Mips),
        9 => Some(Arch::S370),
        10 => Some(Arch::MipsRs3Le),
        15 => Some(Arch::PaRisc),
        17 => Some(Arch::Vpp500),
        18 => Some(Arch::Sparc32Plus),
        19 => Some(Arch::I960),
        20 => Some(Arch::Ppc),
        21 => Some(Arch::Ppc64),
        22 => Some(Arch::S390),
        // 23-35 reserved
        36 => Some(Arch::V800),
        37 => Some(Arch::Fr20),
        38 => Some(Arch::Rh32),
        39 => Some(Arch::Rce),
        40 => Some(Arch::Arm),
        41 => Some(Arch::Alpha),
        42 => Some(Arch::Sh),
        43 => Some(Arch::SparcV9),
        44 => Some(Arch::Tricore),
        45 => Some(Arch::Arc),
        46 => Some(Arch::H8300),
        47 => Some(Arch::H8300H),
        48 => Some(Arch::H8S),
        49 => Some(Arch::H8500),
        50 => Some(Arch::Ia64),
        51 => Some(Arch::MipsX),
        52 => Some(Arch::ColdFire),
        53 => Some(Arch::M68Hc12),
        54 => Some(Arch::Mma),
        55 => Some(Arch::Pcp),
        56 => Some(Arch::Ncpu),
        57 => Some(Arch::Ndr1),
        58 => Some(Arch::StarCore),
        59 => Some(Arch::Me16),
        60 => Some(Arch::St100),
        61 => Some(Arch::TinyJ),
        62 => Some(Arch::X86_64),
        63 => Some(Arch::Pdsp),
        64 => Some(Arch::Pdp10),
        65 => Some(Arch::Pdp11),
        66 => Some(Arch::Fx66),
        67 => Some(Arch::St9Plus),
        68 => Some(Arch::St7),
        69 => Some(Arch::M68Hc16),
        70 => Some(Arch::M68Hc11),
        71 => Some(Arch::M68Hc08),
        72 => Some(Arch::M68Hc05),
        73 => Some(Arch::Svx),
        74 => Some(Arch::St19),
        75 => Some(Arch::Vax),
        76 => Some(Arch::Cris),
        77 => Some(Arch::Javelin),
        78 => Some(Arch::FirePath),
        79 => Some(Arch::Zsp),
        80 => Some(Arch::Mmix),
        81 => Some(Arch::Huany),
        82 => Some(Arch::Prism),
        83 => Some(Arch::Avr),
        84 => Some(Arch::Fr30),
        85 => Some(Arch::D10V),
        86 => Some(Arch::D30V),
        87 => Some(Arch::V850),
        88 => Some(Arch::M32R),
        89 => Some(Arch::Mn10300),
        90 => Some(Arch::Mn10200),
        91 => Some(Arch::PicoJava),
        92 => Some(Arch::OpenRisc),
        93 => Some(Arch::ArcA5),
        94 => Some(Arch::Xtensa),
        95 => Some(Arch::VideoCore),
        96 => Some(Arch::TmmGpp),
        97 => Some(Arch::Ns32K),
        98 => Some(Arch::Tpc),
        99 => Some(Arch::Snp1K),
        100 => Some(Arch::St200),
        106 => Some(Arch::Blackfin),
        110 => Some(Arch::Unicore),
        113 => Some(Arch::AlteraNios2),
        140 => Some(Arch::TIC6000),
        164 => Some(Arch::Hexagon),
        167 => Some(Arch::NDS32),
        183 => Some(Arch::AArch64),
        188 => Some(Arch::TILEPro),
        189 => Some(Arch::Microblaze),
        191 => Some(Arch::TILEGx),
        195 => Some(Arch::ARCv2),
        243 => Some(Arch::RISCV),
        247 => Some(Arch::BPF),
        252 => Some(Arch::CSKY),
        258 => Some(Arch::LoongArch),
        0x5441 => Some(Arch::FRV),
        _ => None,
    }
}
