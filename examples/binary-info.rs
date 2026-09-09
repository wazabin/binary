//! `binary-info` — load a binary and report its container/format metadata.
//!
//! Arguments come either from flags or from one JSON object whose keys are
//! the flag names; `--json` switches the output to JSON.
//!
//! ```text
//! binary-info /bin/ls --sections --symbols
//! binary-info --args '{"path":"/bin/ls","symbols":true}' --json
//! binary-info /bin/ls --read 0x401000:16
//! ```

use std::{fs, process};

use clap::Parser;
use serde::{Deserialize, Serialize};
use wazabin_binary::{Arch, BinaryFormat, blob::Blob, elf::ElfBinary, pe::PeBinary};

/// Load a binary and report its container/format metadata.
#[derive(Parser)]
#[command(name = "binary-info", version)]
struct Cli {
    /// All options as one JSON object; keys are the long flag names
    #[arg(long, value_name = "JSON", conflicts_with_all = ["path", "sections", "symbols", "read", "symbol", "base"])]
    args: Option<String>,
    /// Emit JSON instead of text
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    opts: Opts,
}

/// The options proper: one struct for both the flags and the `--args` JSON.
#[derive(clap::Args, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct Opts {
    /// Path to the binary to load
    path: Option<String>,
    /// Include the section/segment table
    #[arg(long)]
    sections: bool,
    /// Include the symbol table (can be long)
    #[arg(long)]
    symbols: bool,
    /// Print the bytes at ADDR as hex; format is ADDR:SIZE (decimal or 0x-prefixed)
    #[arg(long, value_name = "ADDR:SIZE")]
    read: Option<String>,
    /// Look up one symbol's address by name
    #[arg(long)]
    symbol: Option<String>,
    /// Load address for a raw blob when the file is not ELF or PE (decimal or 0x-prefixed)
    #[arg(long, default_value = "0")]
    #[serde(default = "default_base")]
    base: String,
}

fn default_base() -> String {
    "0".to_string()
}

#[derive(Serialize)]
struct Section {
    name: String,
    address: String,
    size: u64,
    flags: String,
}

#[derive(Serialize)]
struct Symbol {
    name: String,
    address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
}

#[derive(Serialize)]
struct Output {
    path: String,
    format: String,
    arch: String,
    bits: u32,
    endian: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    entry: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sections: Option<Vec<Section>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbols: Option<Vec<Symbol>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_lookup: Option<SymbolLookup>,
    #[serde(skip_serializing_if = "Option::is_none")]
    read: Option<ReadResult>,
}

#[derive(Serialize)]
struct SymbolLookup {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    address: Option<String>,
}

#[derive(Serialize)]
struct ReadResult {
    address: String,
    size: usize,
    hex: String,
}

/// The three container kinds this crate can parse, unified behind
/// [`BinaryFormat`] for the address-space queries and matched on directly for
/// the format-specific section/symbol tables (the trait does not expose
/// those).
enum Loaded {
    Elf(ElfBinary),
    Pe(PeBinary),
    Blob(Blob),
}

impl Loaded {
    fn as_binary_format(&self) -> &dyn BinaryFormat {
        match self {
            Loaded::Elf(b) => b,
            Loaded::Pe(b) => b,
            Loaded::Blob(b) => b,
        }
    }

    fn format_name(&self) -> &'static str {
        match self {
            Loaded::Elf(_) => "elf",
            Loaded::Pe(_) => "pe",
            Loaded::Blob(_) => "blob",
        }
    }

    fn bits(&self) -> u32 {
        match self {
            Loaded::Elf(b) => bits_for_arch(b.architecture),
            Loaded::Pe(b) => {
                if b.is_64 {
                    64
                } else {
                    32
                }
            }
            Loaded::Blob(_) => 64,
        }
    }

    /// This crate does not record the byte order it parsed a container with
    /// (ELF's `AnyEndian` is consumed internally, and PE/blob have no field
    /// for it), so this is a guess from the architecture rather than a fact
    /// read from the file. See the API-gaps note in the accompanying report.
    fn endian(&self) -> &'static str {
        match self {
            Loaded::Elf(b) => endian_guess_for_arch(b.architecture),
            Loaded::Pe(_) => "little",
            Loaded::Blob(_) => "little",
        }
    }

    fn sections(&self) -> Vec<Section> {
        match self {
            Loaded::Elf(b) => b
                .segments
                .iter()
                .enumerate()
                .map(|(i, seg)| Section {
                    name: format!("LOAD{i}"),
                    address: format!("{:#x}", seg.start),
                    size: seg.mem_size,
                    flags: format!(
                        "r{}{}",
                        if seg.writable { "w" } else { "-" },
                        if seg.executable { "x" } else { "-" }
                    ),
                })
                .collect(),
            // `PeSection` records only the writable bit: the loader marks
            // every mapped PE section executable by default (see
            // `PeBinary::mapped_regions`), so the `x` flag here is not read
            // from `IMAGE_SCN_MEM_EXECUTE` and is only an approximation.
            Loaded::Pe(b) => b
                .sections
                .iter()
                .enumerate()
                .map(|(i, sec)| Section {
                    name: format!("SECTION{i}"),
                    address: format!("{:#x}", sec.start),
                    size: sec.mem_size,
                    flags: format!("r{}x", if sec.writable { "w" } else { "-" }),
                })
                .collect(),
            Loaded::Blob(b) => vec![Section {
                name: "blob".to_string(),
                address: format!("{:#x}", b.load_address),
                size: b.data.len() as u64,
                flags: "rwx".to_string(),
            }],
        }
    }

    fn symbols(&self) -> Vec<Symbol> {
        match self {
            // Neither ELF nor PE symbol collection in this crate records a
            // symbol's size, so `size` is always `None` here.
            Loaded::Elf(b) => b
                .analysis
                .known_functions
                .iter()
                .filter_map(|f| {
                    f.name.as_ref().map(|name| Symbol {
                        name: name.clone(),
                        address: format!("{:#x}", f.address),
                        size: None,
                    })
                })
                .collect(),
            Loaded::Pe(b) => b
                .analysis
                .known_functions
                .iter()
                .filter_map(|f| {
                    f.name.as_ref().map(|name| Symbol {
                        name: name.clone(),
                        address: format!("{:#x}", f.address),
                        size: None,
                    })
                })
                .collect(),
            Loaded::Blob(_) => Vec::new(),
        }
    }

    fn find_symbol(&self, name: &str) -> Option<u64> {
        match self {
            Loaded::Elf(b) => b
                .analysis
                .known_functions
                .iter()
                .find(|f| f.name.as_deref() == Some(name))
                .map(|f| f.address),
            Loaded::Pe(b) => b
                .analysis
                .known_functions
                .iter()
                .find(|f| f.name.as_deref() == Some(name))
                .map(|f| f.address),
            Loaded::Blob(_) => None,
        }
    }
}

fn bits_for_arch(arch: Arch) -> u32 {
    match arch {
        Arch::X86_64
        | Arch::AArch64
        | Arch::Ppc64
        | Arch::Ia64
        | Arch::SparcV9
        | Arch::Alpha
        | Arch::Mmix
        | Arch::TILEGx
        | Arch::LoongArch => 64,
        _ => 32,
    }
}

fn endian_guess_for_arch(arch: Arch) -> &'static str {
    match arch {
        Arch::Sparc | Arch::SparcV9 | Arch::S390 | Arch::M68K | Arch::Mips | Arch::MipsRs3Le => {
            "big"
        }
        _ => "little",
    }
}

fn parse_int(what: &str, value: &str) -> Result<u64, String> {
    let value = value.trim();
    match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|_| format!("invalid {what} '{value}'"))
}

fn parse_read_arg(value: &str) -> Result<(u64, usize), String> {
    let (addr, size) = value
        .split_once(':')
        .ok_or_else(|| format!("--read expects ADDR:SIZE, got '{value}'"))?;
    let addr = parse_int("--read address", addr)?;
    let size = parse_int("--read size", size)? as usize;
    Ok((addr, size))
}

fn load(path: &str, base: u64) -> Result<Loaded, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() >= 4 && &bytes[0..4] == b"\x7fELF" {
        return ElfBinary::parse(&bytes)
            .map(Loaded::Elf)
            .map_err(|e| format!("failed to parse ELF: {e}"));
    }
    if bytes.len() >= 2 && &bytes[0..2] == b"MZ" {
        return PeBinary::parse(&bytes)
            .map(Loaded::Pe)
            .map_err(|e| format!("failed to parse PE: {e}"));
    }
    Ok(Loaded::Blob(Blob::new(base, bytes)))
}

fn run(opts: &Opts) -> Result<Output, String> {
    let path = opts.path.clone().ok_or_else(|| "give a path".to_string())?;
    let base = parse_int("base", &opts.base)?;
    let loaded = load(&path, base)?;
    let binary = loaded.as_binary_format();

    let entry = binary.entrypoint().map(|a| format!("{a:#x}"));
    let sections = opts.sections.then(|| loaded.sections());
    let symbols = opts.symbols.then(|| loaded.symbols());
    let symbol_lookup = opts.symbol.as_ref().map(|name| SymbolLookup {
        name: name.clone(),
        address: loaded.find_symbol(name).map(|a| format!("{a:#x}")),
    });
    let read = match &opts.read {
        Some(spec) => {
            let (addr, size) = parse_read_arg(spec)?;
            let bytes = binary
                .read_bytes(addr, size)
                .ok_or_else(|| format!("cannot read {size} byte(s) at {addr:#x}: unmapped"))?;
            Some(ReadResult {
                address: format!("{addr:#x}"),
                size,
                hex: bytes.iter().map(|b| format!("{b:02x}")).collect(),
            })
        }
        None => None,
    };

    Ok(Output {
        path,
        format: loaded.format_name().to_string(),
        arch: binary.architecture().to_string(),
        bits: loaded.bits(),
        endian: loaded.endian().to_string(),
        entry,
        sections,
        symbols,
        symbol_lookup,
        read,
    })
}

fn main() {
    let cli = Cli::parse();
    let opts = match &cli.args {
        Some(json) => match serde_json::from_str::<Opts>(json) {
            Ok(opts) => opts,
            Err(e) => fail(cli.json, &format!("invalid --args: {e}")),
        },
        None => cli.opts,
    };
    let output = match run(&opts) {
        Ok(output) => output,
        Err(e) => fail(cli.json, &e),
    };
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }

    println!("path:   {}", output.path);
    println!("format: {}", output.format);
    println!("arch:   {} ({}-bit, {}-endian)", output.arch, output.bits, output.endian);
    println!("entry:  {}", output.entry.as_deref().unwrap_or("-"));

    if let Some(sections) = &output.sections {
        println!("\nsections:");
        println!("{:<12} {:<12} {:<10} {:<6}", "name", "address", "size", "flags");
        for s in sections {
            println!("{:<12} {:<12} {:<10} {:<6}", s.name, s.address, s.size, s.flags);
        }
    }

    if let Some(symbols) = &output.symbols {
        println!("\nsymbols ({}):", symbols.len());
        println!("{:<12} {}", "address", "name");
        for s in symbols {
            println!("{:<12} {}", s.address, s.name);
        }
    }

    if let Some(lookup) = &output.symbol_lookup {
        match &lookup.address {
            Some(addr) => println!("\nsymbol {}: {addr}", lookup.name),
            None => println!("\nsymbol {}: not found", lookup.name),
        }
    }

    if let Some(read) = &output.read {
        println!("\nread {} byte(s) at {}:", read.size, read.address);
        println!("{}", read.hex);
    }
}

fn fail(json: bool, message: &str) -> ! {
    if json {
        eprintln!("{}", serde_json::json!({ "error": message }));
    } else {
        eprintln!("error: {message}");
    }
    process::exit(1);
}
