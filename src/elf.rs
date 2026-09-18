use elf::{
    ElfBytes,
    abi::{
        DT_NEEDED, EM_386, EM_X86_64, ET_CORE, ET_DYN, ET_EXEC, ET_REL, PF_W, PF_X, PT_LOAD,
        PT_PHDR, PT_TLS, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, SHF_ALLOC, SHF_EXECINSTR,
        SHF_WRITE, SHT_DYNSYM, SHT_NOBITS, SHT_REL, SHT_RELA, SHT_SYMTAB, STT_FUNC,
    },
    endian::{AnyEndian, EndianParse},
};

use crate::{Arch, BinaryFormat, Endian, Section, Symbol};

use crate::symbols::SymbolIndex;

/// A known function start address extracted from an ELF symbol table.
#[derive(Debug, Clone)]
pub struct FunctionSymbol {
    /// Virtual address of the function.
    pub address: u64,
    /// Name from the string table, if present.
    pub name: Option<String>,
    /// `st_size` when the symbol table records a non-zero one. `None` for
    /// PLT stubs, `.eh_frame` starts, and assembly symbols with no `.size`.
    pub size: Option<u64>,
    /// Whether this is an external (imported) function stub (e.g. a PLT thunk).
    pub is_external: bool,
    /// For an external stub, the shared library its symbol-version requirement
    /// (`.gnu.version_r`) names, e.g. `libc.so.6`. `None` for unversioned
    /// imports (the format does not tie those to a specific `DT_NEEDED` entry).
    pub library: Option<String>,
}

/// A resolver slot for an imported function.
#[derive(Debug, Clone)]
pub struct ImportSymbol {
    /// Virtual address of the GOT / PLT relocation slot.
    pub address: u64,
    /// Imported function name.
    pub name: String,
    /// The shared library the symbol's version requirement (`.gnu.version_r`)
    /// names, e.g. `libc.so.6`. `None` for unversioned imports.
    pub library: Option<String>,
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
    /// The segment's alignment (`p_align`); zero or one means none.
    pub align: u64,
}

/// An allocated section header (`SHF_ALLOC` set): a named slice of the
/// loaded image such as `.text` or `.bss`. Absent from a file whose section
/// table was stripped, which loaders never need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfSection {
    /// The name from `.shstrtab`.
    pub name: String,
    /// `sh_addr`.
    pub address: u64,
    /// `sh_size`.
    pub size: u64,
    /// `SHF_WRITE`.
    pub writable: bool,
    /// `SHF_EXECINSTR`.
    pub executable: bool,
    /// `SHT_NOBITS`: the section occupies no file bytes and is zero-filled
    /// by the loader (`.bss`, `.tbss`).
    pub nobits: bool,
}

/// The file type from `e_type`: what kind of object this is, and so whether
/// a loader may place it where it likes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfKind {
    /// `ET_EXEC`: linked to run at its segments' addresses.
    Executable,
    /// `ET_DYN`: a shared object or a position-independent executable; the
    /// loader chooses the base and adds it to every address.
    SharedObject,
    /// `ET_REL`: an unlinked object file.
    Relocatable,
    /// `ET_CORE`: a core dump.
    Core,
    /// Anything else, with the raw `e_type`.
    Other(u16),
}

impl ElfKind {
    fn from_e_type(value: u16) -> Self {
        match value {
            ET_EXEC => ElfKind::Executable,
            ET_DYN => ElfKind::SharedObject,
            ET_REL => ElfKind::Relocatable,
            ET_CORE => ElfKind::Core,
            other => ElfKind::Other(other),
        }
    }
}

/// Where the program header table is, in the file and, when a segment
/// covers it, in memory. A loader passes the virtual address to the program
/// as `AT_PHDR`; libc startup and the dynamic linker walk the table from
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramHeaderTable {
    /// `e_phoff`.
    pub offset: u64,
    /// `e_phentsize`.
    pub entry_size: u16,
    /// `e_phnum`.
    pub count: u16,
    /// The table's unrelocated virtual address: what `PT_PHDR` names, else
    /// the address inside the `PT_LOAD` segment whose file bytes cover
    /// `offset`. `None` when no segment maps it.
    pub vaddr: Option<u64>,
}

/// The `PT_TLS` segment: the thread-local storage template a runtime copies
/// for each thread. At most one per file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsSegment {
    /// Unrelocated virtual address of the template.
    pub vaddr: u64,
    /// Bytes of the template that come from the file (`.tdata`).
    pub file_size: u64,
    /// Total size, the tail being zero-filled (`.tbss`).
    pub mem_size: u64,
    /// `p_align`.
    pub align: u64,
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
///   `.dynsym`, PLT stubs, and every `.eh_frame` FDE start (nameless, but
///   they survive `strip`).
///
/// `known_functions` and `imported_symbols` are sorted by address with one
/// entry per address, and a name index is built alongside them, which is
/// what the [`BinaryFormat::symbol_name`] / [`BinaryFormat::symbol_address`]
/// lookups search. Both hold once parsing returns; mutating the tables
/// afterwards is not supported.
#[derive(Debug, Clone)]
pub struct ElfBinary {
    pub load_address: u64,
    pub segments: Vec<LoadSegment>,
    pub analysis: ElfAnalysis,
    /// Function names -> addresses, over `analysis.known_functions`.
    symbols: SymbolIndex,
    pub architecture: Arch,
    /// `EI_DATA`: the byte order every multi-byte field was read with.
    pub endian: Endian,
    /// `EI_CLASS`: `true` for ELF64, `false` for ELF32.
    pub is_64: bool,
    /// The allocated section headers in address order, empty when the
    /// section table was stripped. What [`BinaryFormat::sections`] reports.
    pub sections: Vec<ElfSection>,
    /// Shared-library sonames from the `.dynamic` section's `DT_NEEDED` entries,
    /// in link order (e.g. `libc.so.6`).
    pub needed_libraries: Vec<String>,
    /// The file type.
    pub kind: ElfKind,
    /// The program header table's location.
    pub program_headers: ProgramHeaderTable,
    /// The thread-local storage template, if the file has one.
    pub tls: Option<TlsSegment>,
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
                    align: p.p_align,
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

        // --- Analysis pass 6: `.eh_frame` FDE starts ---------------------------
        // Pushed last so a named symbol at the same address wins the dedup
        // below. Skipped for the PLT stub tables (their FDE covers the whole
        // table, and the stubs are already listed by name) and for anything
        // outside executable memory (a stale FDE in a hand-edited binary).
        let plt_ranges = plt_section_ranges(&elf);
        for addr in crate::eh_frame::function_starts(&elf, file_bytes) {
            let in_plt = plt_ranges.iter().any(|(s, e)| addr >= *s && addr < *e);
            let executable = load_segments
                .iter()
                .any(|seg| seg.executable && seg.contains(addr));
            if !in_plt && executable {
                known_functions.push(FunctionSymbol {
                    address: addr,
                    name: None,
                    size: None,
                    is_external: false,
                    library: None,
                });
            }
        }

        // --- Analysis pass 7: constructor / destructor entry points -----------
        // The crt functions `frame_dummy` and `__do_global_dtors_aux` have no
        // FDE, but the loader reaches them through `.init_array` /
        // `.fini_array` (and `DT_INIT` / `DT_FINI`), so those pointer tables
        // are function starts too.
        for addr in init_fini_entries(&elf, &load_segments) {
            let executable = load_segments
                .iter()
                .any(|seg| seg.executable && seg.contains(addr));
            if executable {
                known_functions.push(FunctionSymbol {
                    address: addr,
                    name: None,
                    size: None,
                    is_external: false,
                    library: None,
                });
            }
        }

        // --- The section table, for callers that want names -------------------
        let sections = collect_sections(&elf);

        // --- What a loader needs beyond the segments ---------------------------
        let endian = if elf.ehdr.endianness.is_little() {
            Endian::Little
        } else {
            Endian::Big
        };
        let is_64 = matches!(elf.ehdr.class, elf::file::Class::ELF64);
        let kind = ElfKind::from_e_type(elf.ehdr.e_type);
        let phoff = elf.ehdr.e_phoff;
        let phdr_vaddr = segments
            .iter()
            .find(|p| p.p_type == PT_PHDR)
            .map(|p| p.p_vaddr)
            .or_else(|| {
                segments
                    .iter()
                    .filter(|p| p.p_type == PT_LOAD)
                    .find(|p| phoff >= p.p_offset && phoff < p.p_offset + p.p_filesz)
                    .map(|p| p.p_vaddr + (phoff - p.p_offset))
            });
        let program_headers = ProgramHeaderTable {
            offset: phoff,
            entry_size: elf.ehdr.e_phentsize,
            count: elf.ehdr.e_phnum,
            vaddr: phdr_vaddr,
        };
        let tls = segments
            .iter()
            .find(|p| p.p_type == PT_TLS)
            .map(|p| TlsSegment {
                vaddr: p.p_vaddr,
                file_size: p.p_filesz,
                mem_size: p.p_memsz,
                align: p.p_align,
            });

        // Deduplicate by address (symtab may overlap dynsym). The sorted
        // order is what `function_at` / `import_at` binary-search.
        known_functions.sort_by_key(|f| f.address);
        known_functions.dedup_by_key(|f| f.address);
        imported_symbols.sort_by_key(|f| f.address);
        imported_symbols.dedup_by_key(|f| f.address);
        let symbols = SymbolIndex::build(
            known_functions
                .iter()
                .map(|f| (f.name.as_deref(), f.address, f.is_external)),
        );

        Ok(ElfBinary {
            load_address,
            segments: load_segments,
            analysis: ElfAnalysis {
                entrypoint,
                known_functions,
                imported_symbols,
            },
            symbols,
            architecture: from_elf_machine(elf.ehdr.e_machine).ok_or(ElfError::UnknownArch)?,
            endian,
            is_64,
            sections,
            needed_libraries,
            kind,
            program_headers,
            tls,
        })
    }

    /// The known function starting at `addr`, by binary search over the
    /// address-sorted table.
    fn function_at(&self, addr: u64) -> Option<&FunctionSymbol> {
        let functions = &self.analysis.known_functions;
        let idx = functions.binary_search_by_key(&addr, |f| f.address).ok()?;
        Some(&functions[idx])
    }

    /// The imported symbol whose resolver slot is at `addr`, by binary search
    /// over the address-sorted table.
    fn import_at(&self, addr: u64) -> Option<&ImportSymbol> {
        let imports = &self.analysis.imported_symbols;
        let idx = imports.binary_search_by_key(&addr, |f| f.address).ok()?;
        Some(&imports[idx])
    }
}

/// The `SHF_ALLOC` section headers with a name, in address order. Empty when
/// the file has no section table or no `.shstrtab`.
fn collect_sections(elf: &ElfBytes<AnyEndian>) -> Vec<ElfSection> {
    let (shdrs, shstrtab) = match elf.section_headers_with_strtab() {
        Ok((Some(s), Some(st))) => (s, st),
        _ => return Vec::new(),
    };
    let mut out: Vec<ElfSection> = shdrs
        .iter()
        .filter(|s| s.sh_flags & u64::from(SHF_ALLOC) != 0)
        .filter_map(|s| {
            let name = shstrtab.get(s.sh_name as usize).ok()?;
            Some(ElfSection {
                name: name.to_owned(),
                address: s.sh_addr,
                size: s.sh_size,
                writable: s.sh_flags & u64::from(SHF_WRITE) != 0,
                executable: s.sh_flags & u64::from(SHF_EXECINSTR) != 0,
                nobits: s.sh_type == SHT_NOBITS,
            })
        })
        .collect();
    out.sort_by_key(|s| s.address);
    out
}

/// `[start, end)` virtual ranges of the PLT stub tables (`.plt`, `.plt.sec`,
/// `.plt.got`), for excluding their FDEs from function discovery.
fn plt_section_ranges(elf: &ElfBytes<AnyEndian>) -> Vec<(u64, u64)> {
    let (shdrs, shstrtab) = match elf.section_headers_with_strtab() {
        Ok((Some(s), Some(st))) => (s, st),
        _ => return Vec::new(),
    };
    shdrs
        .iter()
        .filter(|s| {
            matches!(
                shstrtab.get(s.sh_name as usize).ok(),
                Some(".plt" | ".plt.sec" | ".plt.got")
            )
        })
        .map(|s| (s.sh_addr, s.sh_addr + s.sh_size))
        .collect()
}

/// Function pointers stored in `.preinit_array` / `.init_array` /
/// `.fini_array`, plus the `DT_INIT` / `DT_FINI` dynamic entries. Read from
/// the loaded image, so a non-PIE binary yields absolute addresses and a PIE
/// (whose slots hold `R_X86_64_RELATIVE` addends) its link-time ones.
fn init_fini_entries(elf: &ElfBytes<AnyEndian>, segments: &[LoadSegment]) -> Vec<u64> {
    let ptr_size = match elf.ehdr.class {
        elf::file::Class::ELF32 => 4,
        elf::file::Class::ELF64 => 8,
    };
    let read_ptr = |addr: u64| -> Option<u64> {
        let seg = segments.iter().find(|s| s.contains(addr))?;
        let bytes = seg.bytes_at(addr)?;
        match ptr_size {
            4 => Some(u64::from(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))),
            _ => Some(u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?)),
        }
    };
    let mut out = Vec::new();
    if let Ok((Some(shdrs), Some(shstrtab))) = elf.section_headers_with_strtab() {
        for shdr in shdrs.iter() {
            if !matches!(
                shstrtab.get(shdr.sh_name as usize).ok(),
                Some(".preinit_array" | ".init_array" | ".fini_array")
            ) {
                continue;
            }
            let mut slot = shdr.sh_addr;
            let end = shdr.sh_addr.saturating_add(shdr.sh_size);
            while slot + ptr_size as u64 <= end {
                match read_ptr(slot) {
                    // `0` and `-1` are the linker's empty / sentinel slots.
                    Some(0) | Some(u64::MAX) | None => {}
                    Some(target) => out.push(target),
                }
                slot += ptr_size as u64;
            }
        }
    }
    if let Ok(Some(dynamic)) = elf.dynamic() {
        for entry in dynamic.iter() {
            if (entry.d_tag == elf::abi::DT_INIT || entry.d_tag == elf::abi::DT_FINI)
                && entry.d_ptr() != 0
            {
                out.push(entry.d_ptr());
            }
        }
    }
    out
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
                    size: (sym.st_size != 0).then_some(sym.st_size),
                    is_external: false,
                    library: None,
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
                    size: (sym.st_size != 0).then_some(sym.st_size),
                    is_external: false,
                    library: None,
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
    // Symbol-version table (`.gnu.version` / `.gnu.version_r`): maps a dynsym
    // index to the shared library its version requirement names. Absent in
    // unversioned binaries, in which case imports carry no library.
    let version_table = elf.symbol_version_table().ok().flatten();
    let import_library = |sym_idx: u32| {
        version_table
            .as_ref()
            .and_then(|vt| vt.get_requirement(sym_idx as usize).ok().flatten())
            .map(|req| req.file.to_string())
    };

    // Import relocations from `.rela.plt` (x86-64 / RELA) or `.rel.plt`
    // (x86-32 / REL), as `(index, symbol_name, library)` tuples in section order.
    let mut import_relocs: Vec<(usize, Option<String>, Option<String>)> = Vec::new();
    if let Some(rela_plt_shdr) = shdrs
        .iter()
        .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(".rela.plt"))
    {
        for (i, rela) in elf.section_data_as_relas(&rela_plt_shdr).ok()?.enumerate() {
            if is_import_relocation(elf.ehdr.e_machine, rela.r_type) {
                let name = dyn_symbol_name(&dynsymtab, &dynstrtab, rela.r_sym).map(str::to_string);
                import_relocs.push((i, name, import_library(rela.r_sym)));
            }
        }
    } else if let Some(rel_plt_shdr) = shdrs
        .iter()
        .find(|s| shstrtab.get(s.sh_name as usize).ok() == Some(".rel.plt"))
    {
        for (i, rel) in elf.section_data_as_rels(&rel_plt_shdr).ok()?.enumerate() {
            if is_import_relocation(elf.ehdr.e_machine, rel.r_type) {
                let name = dyn_symbol_name(&dynsymtab, &dynstrtab, rel.r_sym).map(str::to_string);
                import_relocs.push((i, name, import_library(rel.r_sym)));
            }
        }
    }

    for (i, name, library) in import_relocs {
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
                    size: None,
                    is_external: true,
                    library: library.clone(),
                });
                out.push(FunctionSymbol {
                    address: plt_stub_addr,
                    name: None,
                    size: None,
                    is_external: true,
                    library,
                });
            }
            None => out.push(FunctionSymbol {
                address: plt_stub_addr,
                name,
                size: None,
                is_external: true,
                library,
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
    let version_table = elf.symbol_version_table().ok().flatten();
    let import_library = |sym_idx: u32| {
        version_table
            .as_ref()
            .and_then(|vt| vt.get_requirement(sym_idx as usize).ok().flatten())
            .map(|req| req.file.to_string())
    };

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
                        library: import_library(rela.r_sym),
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
                        library: import_library(rel.r_sym),
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

    fn endianness(&self) -> Endian {
        self.endian
    }

    fn bits(&self) -> u32 {
        if self.is_64 { 64 } else { 32 }
    }

    /// The allocated section headers, or, for a file whose section table
    /// was stripped, the `PT_LOAD` segments named `LOAD0`, `LOAD1`, … so the
    /// layout is still enumerable.
    fn sections(&self) -> Vec<Section> {
        if !self.sections.is_empty() {
            return self
                .sections
                .iter()
                .map(|s| Section {
                    name: s.name.clone(),
                    address: s.address,
                    size: s.size,
                    writable: s.writable,
                    executable: s.executable,
                })
                .collect();
        }
        self.segments
            .iter()
            .enumerate()
            .map(|(i, seg)| Section {
                name: format!("LOAD{i}"),
                address: seg.start,
                size: seg.mem_size,
                writable: seg.writable,
                executable: seg.executable,
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

    fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.segments
            .iter()
            .find(|segment| segment.contains(addr))
            .map(|segment| (segment.start, segment.end()))
    }

    fn is_known_writable(&self, addr: u64) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.writable && segment.contains(addr))
    }

    /// ELF program headers carry `PF_W`, so a mapped segment without it is
    /// proven read-only for the lifetime of the process.
    fn is_known_read_only(&self, addr: u64) -> bool {
        self.segments
            .iter()
            .any(|segment| !segment.writable && segment.contains(addr))
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
        self.function_at(addr)
            .and_then(|f| f.name.as_deref())
            .filter(|name| !name.is_empty())
            .or_else(|| {
                (addr == self.analysis.entrypoint && self.is_executable(addr)).then_some("_start")
            })
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

    fn import_symbol_name(&self, addr: u64) -> Option<&str> {
        self.import_at(addr).map(|f| f.name.as_str())
    }

    fn import_library(&self, addr: u64) -> Option<&str> {
        // Externals are minted either at a PLT stub's address or (for GOT-
        // indirect calls with no PLT stub) at the relocation slot itself.
        self.function_at(addr)
            .and_then(|f| f.library.as_deref())
            .or_else(|| self.import_at(addr).and_then(|f| f.library.as_deref()))
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
