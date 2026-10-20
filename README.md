# wazabin-binary

Binary container-format abstractions and architecture metadata for binary
analysis tools. It provides the `BinaryFormat` trait and implementations or
supporting types for blob, ELF, and PE inputs.

## Usage

```rust
use wazabin_binary::{BinaryFormat, Format, load};

let bytes = std::fs::read("/bin/ls")?;
let binary = load(&bytes)?; // Box<dyn BinaryFormat>, ELF or PE by magic

println!("{:?} {}-bit {:?}", Format::detect(&bytes), binary.bits(), binary.endianness());
for section in binary.sections() {
    println!("{:<20} {:#x} {}", section.name, section.address, section.size);
}
if let Some(main) = binary.symbol_address("main") {
    println!("main at {main:#x}");
}
```

Match on `Format::detect` and call `elf::ElfBinary::parse` or `pe::PeBinary::parse`
directly to keep the concrete type and its format-specific tables.

## The loader view

`image::Image` is a borrowed, one-pass view of an ELF file: segments,
sections (empty without section headers; none are invented), relocations,
symbols, imports with their PLT stubs, the dynamic table and TLS. Nothing is
copied out of the input, and `.eh_frame` is never parsed at load time.

```rust
use wazabin_binary::image::Image;

let bytes = std::fs::read("/bin/ls")?;
let img = Image::load(&bytes)?; // Image::load_segments(&bytes): no section headers read
for import in &img.imports {
    println!("{:<24} slot {:?} stub {:?}", import.name, import.slot, import.stub);
}
// The exception tables (CIE / FDE / LSDA) are read on first use.
println!("{} functions with an LSDA", img.eh().funcs.len());
// Function starts the metadata names, apart from the symbols.
let seeds = img.seeds();
```
