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
use wazabin_binary::{BinaryFormat, Endian, Format, LoadError, blob::Blob};

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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    external: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    library: Option<String>,
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

/// A parsed file: the container the crate detected, or a raw blob when it
/// detected none, behind the one trait every query below goes through.
struct Loaded {
    format: Option<Format>,
    binary: Box<dyn BinaryFormat>,
}

impl Loaded {
    fn format_name(&self) -> String {
        self.format
            .map_or_else(|| "blob".to_string(), |f| f.to_string())
    }

    fn sections(&self) -> Vec<Section> {
        self.binary
            .sections()
            .into_iter()
            .map(|s| Section {
                name: s.name,
                address: format!("{:#x}", s.address),
                size: s.size,
                flags: format!(
                    "r{}{}",
                    if s.writable { "w" } else { "-" },
                    if s.executable { "x" } else { "-" }
                ),
            })
            .collect()
    }

    fn symbols(&self) -> Vec<Symbol> {
        self.binary
            .symbols()
            .into_iter()
            .map(|s| Symbol {
                name: s.name,
                address: format!("{:#x}", s.address),
                size: s.size,
                external: s.is_external,
                library: s.library,
            })
            .collect()
    }
}

fn parse_int(what: &str, value: &str) -> Result<u64, String> {
    let value = value.trim();
    match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
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
    let format = Format::detect(&bytes);
    let binary: Box<dyn BinaryFormat> = match wazabin_binary::load(&bytes) {
        Ok(binary) => binary,
        Err(LoadError::UnknownFormat) => Box::new(Blob::new(base, bytes)),
        Err(e) => return Err(e.to_string()),
    };
    Ok(Loaded { format, binary })
}

fn run(opts: &Opts) -> Result<Output, String> {
    let path = opts.path.clone().ok_or_else(|| "give a path".to_string())?;
    let base = parse_int("base", &opts.base)?;
    let loaded = load(&path, base)?;
    let binary = loaded.binary.as_ref();

    let entry = binary.entrypoint().map(|a| format!("{a:#x}"));
    let sections = opts.sections.then(|| loaded.sections());
    let symbols = opts.symbols.then(|| loaded.symbols());
    let symbol_lookup = opts.symbol.as_ref().map(|name| SymbolLookup {
        name: name.clone(),
        address: binary.symbol_address(name).map(|a| format!("{a:#x}")),
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
        format: loaded.format_name(),
        arch: binary.architecture().to_string(),
        bits: binary.bits(),
        endian: match binary.endianness() {
            Endian::Little => "little".to_string(),
            Endian::Big => "big".to_string(),
        },
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
    println!(
        "arch:   {} ({}-bit, {}-endian)",
        output.arch, output.bits, output.endian
    );
    println!("entry:  {}", output.entry.as_deref().unwrap_or("-"));

    if let Some(sections) = &output.sections {
        println!("\nsections:");
        println!(
            "{:<20} {:<12} {:<10} {:<6}",
            "name", "address", "size", "flags"
        );
        for s in sections {
            println!(
                "{:<20} {:<12} {:<10} {:<6}",
                s.name, s.address, s.size, s.flags
            );
        }
    }

    if let Some(symbols) = &output.symbols {
        println!("\nsymbols ({}):", symbols.len());
        println!("{:<12} {:<8} name", "address", "size");
        for s in symbols {
            let size = s.size.map_or_else(|| "-".to_string(), |n| n.to_string());
            let origin = match (&s.external, &s.library) {
                (true, Some(lib)) => format!("  [{lib}]"),
                (true, None) => "  [import]".to_string(),
                (false, _) => String::new(),
            };
            println!("{:<12} {:<8} {}{origin}", s.address, size, s.name);
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
