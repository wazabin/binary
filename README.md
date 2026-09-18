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
