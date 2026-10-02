//! A byte-assembled x86-64 PIE with everything a loader reads: two `PT_LOAD`
//! segments, a dynamic table with relocations, versions and constructor
//! arrays, PLT stubs of three kinds, a TLS template, an `.eh_frame` with a C++
//! style CIE (`zPLR`, indirect personality) and an LSDA, and, optionally, a
//! section table.
//!
//! The layout (file offset == address in the first segment; the second is at
//! file offset 0x1000, address 0x3000):
//!
//! | what               | address |
//! |--------------------|---------|
//! | `.dynsym`          | 0x200   |
//! | `.dynstr`          | 0x300   |
//! | `.gnu.version`     | 0x3a0   |
//! | `.gnu.version_r`   | 0x3c0   |
//! | `.hash`            | 0x3e0   |
//! | `.rela.dyn`        | 0x420   |
//! | `.rela.plt`        | 0x500   |
//! | `.plt` (stub: puts)| 0x600 (0x610) |
//! | `.plt.got`         | 0x640   |
//! | `.plt.sec` (free)  | 0x650   |
//! | canonical `qsort`  | 0x660   |
//! | `.text`            | 0x700   |
//! | `.gcc_except_table`| 0x800   |
//! | `.eh_frame_hdr`    | 0x900   |
//! | `.eh_frame`        | 0x940   |
//! | `.dynamic`         | 0x3000  |
//! | `.got.plt`         | 0x3200  |
//! | `.init_array`      | 0x3300  |
//! | `.tdata`           | 0x3400  |
#![allow(dead_code)]

pub const PUTS_SLOT: u64 = 0x3208;
pub const FREE_SLOT: u64 = 0x3210;
pub const FINALIZE_SLOT: u64 = 0x3220;
pub const TYPEINFO_SLOT: u64 = 0x3228;
pub const PERSONALITY_SLOT: u64 = 0x3230;
pub const ABS_SLOT: u64 = 0x3240;
pub const IREL_SLOT: u64 = 0x3248;
pub const TPOFF_SLOT: u64 = 0x3250;
pub const INIT_ARRAY: u64 = 0x3300;

pub struct Buf(pub Vec<u8>);

impl Buf {
    pub fn put(&mut self, at: usize, bytes: &[u8]) {
        if self.0.len() < at + bytes.len() {
            self.0.resize(at + bytes.len(), 0);
        }
        self.0[at..at + bytes.len()].copy_from_slice(bytes);
    }
    pub fn u16(&mut self, at: usize, v: u16) {
        self.put(at, &v.to_le_bytes());
    }
    pub fn u32(&mut self, at: usize, v: u32) {
        self.put(at, &v.to_le_bytes());
    }
    pub fn u64(&mut self, at: usize, v: u64) {
        self.put(at, &v.to_le_bytes());
    }
}

/// A pc-relative sdata4 word at `at` pointing to `target`.
fn pcrel(b: &mut Buf, at: u64, target: u64) {
    b.u32(at as usize, (target as i64 - at as i64) as i32 as u32);
}

struct Names {
    bytes: Vec<u8>,
}

impl Names {
    fn new() -> Self {
        Names { bytes: vec![0] }
    }
    fn add(&mut self, s: &str) -> u32 {
        let at = self.bytes.len() as u32;
        self.bytes.extend_from_slice(s.as_bytes());
        self.bytes.push(0);
        at
    }
}

/// The image; `sections` adds a section table (and `.symtab`), `strip_dynamic`
/// is not offered: the dynamic table is always there.
pub fn build(sections: bool) -> Vec<u8> {
    let mut b = Buf(vec![0u8; 0x1000]);

    // --- dynstr and dynsym -------------------------------------------------
    let mut dynstr = Names::new();
    let libc = dynstr.add("libc.so.6");
    let names = [
        "puts",
        "__cxa_finalize",
        "_ZTIi",
        "main",
        "qsort",
        "free",
        "__gxx_personality_v0",
    ]
    .map(|n| dynstr.add(n));
    assert!(dynstr.bytes.len() <= 0xa0);
    b.put(0x300, &dynstr.bytes);
    // (name, info, shndx, value, size)
    let syms: [(u32, u8, u16, u64, u64); 7] = [
        (names[0], 0x12, 0, 0, 0),        // puts: global FUNC, undefined
        (names[1], 0x22, 0, 0, 0),        // __cxa_finalize: weak FUNC, undefined
        (names[2], 0x11, 0, 0, 0),        // _ZTIi: global OBJECT, undefined
        (names[3], 0x12, 7, 0x700, 0x20), // main: defined
        (names[4], 0x12, 0, 0x660, 0),    // qsort: undefined, with a canonical PLT value
        (names[5], 0x12, 0, 0, 0),        // free
        (names[6], 0x12, 0, 0, 0),        // __gxx_personality_v0
    ];
    for (i, s) in syms.iter().enumerate() {
        let at = 0x200 + 24 * (i + 1);
        b.u32(at, s.0);
        b.put(at + 4, &[s.1]);
        b.u16(at + 6, s.2);
        b.u64(at + 8, s.3);
        b.u64(at + 16, s.4);
    }
    // .gnu.version: libc version 2 for puts, __cxa_finalize, free.
    for (i, v) in [0u16, 2, 2, 1, 1, 1, 2, 1].iter().enumerate() {
        b.u16(0x3a0 + 2 * i, *v);
    }
    // .gnu.version_r: one Verneed (file = libc) with one Vernaux (other = 2).
    b.u16(0x3c0, 1);
    b.u16(0x3c2, 1);
    b.u32(0x3c4, libc);
    b.u32(0x3c8, 16);
    b.u32(0x3cc, 0);
    b.u16(0x3d6, 2); // vna_other
    b.u32(0x3d8, libc);
    // .hash: one bucket, eight chains.
    b.u32(0x3e0, 1);
    b.u32(0x3e4, 8);

    // --- relocations ----------------------------------------------------------
    let rela = |b: &mut Buf, at: usize, offset: u64, ty: u32, sym: u32, addend: i64| {
        b.u64(at, offset);
        b.u64(at + 8, (u64::from(sym) << 32) | u64::from(ty));
        b.u64(at + 16, addend as u64);
    };
    rela(&mut b, 0x420, INIT_ARRAY, 8, 0, 0x710); // RELATIVE
    rela(&mut b, 0x438, FINALIZE_SLOT, 6, 2, 0); // GLOB_DAT
    rela(&mut b, 0x450, TYPEINFO_SLOT, 6, 3, 0);
    rela(&mut b, 0x468, PERSONALITY_SLOT, 6, 7, 0);
    rela(&mut b, 0x480, ABS_SLOT, 1, 4, 8); // R_X86_64_64 against main
    rela(&mut b, 0x498, IREL_SLOT, 37, 0, 0x720); // IRELATIVE
    rela(&mut b, 0x4b0, TPOFF_SLOT, 18, 0, 0x10); // TPOFF64
    rela(&mut b, 0x500, PUTS_SLOT, 7, 1, 0); // JUMP_SLOT
    rela(&mut b, 0x518, FREE_SLOT, 7, 6, 0);

    // --- PLT stubs ------------------------------------------------------------
    let jmp = |b: &mut Buf, at: u64, slot: u64, prefix: &[u8], tail: &[u8]| {
        let mut p = at as usize;
        b.put(p, prefix);
        p += prefix.len();
        b.put(p, &[0xff, 0x25]);
        b.u32(p + 2, (slot as i64 - (p as i64 + 6)) as i32 as u32);
        b.put(p + 6, tail);
    };
    b.put(
        0x600,
        &[
            0xff, 0x35, 0, 0, 0, 0, 0xff, 0x25, 0, 0, 0, 0, 0x0f, 0x1f, 0x40, 0,
        ],
    ); // PLT0
    jmp(
        &mut b,
        0x610,
        PUTS_SLOT,
        &[],
        &[0x68, 0, 0, 0, 0, 0xe9, 0xe0, 0xff, 0xff, 0xff],
    );
    jmp(&mut b, 0x640, FINALIZE_SLOT, &[], &[0x66, 0x90]); // .plt.got
    jmp(
        &mut b,
        0x650,
        FREE_SLOT,
        &[0xf3, 0x0f, 0x1e, 0xfa, 0xf2],
        &[0x0f, 0x1f, 0x44, 0, 0],
    ); // .plt.sec
    b.put(0x660, &[0x90; 16]);

    // --- text -------------------------------------------------------------------
    b.put(0x700, &[0xc3; 0x60]);

    // --- the LSDA -----------------------------------------------------------------
    // lpstart omitted; type table: indirect | pcrel | sdata4; call sites uleb.
    let lsda = 0x800usize;
    let call_sites = [(0u8, 4u8, 0x10u8, 1u8), (4, 4, 0, 0), (8, 4, 0x20, 0)];
    let cs_len = call_sites.len() * 4;
    let actions = lsda + 5 + cs_len; // after the header and the call sites
    let type_entry = actions + 2;
    let tt_base = type_entry + 4;
    b.put(lsda, &[0xff, 0x9b]);
    b.put(lsda + 2, &[(tt_base - (lsda + 3)) as u8]);
    b.put(lsda + 3, &[0x01, cs_len as u8]);
    for (i, (s, l, lp, a)) in call_sites.iter().enumerate() {
        b.put(lsda + 5 + 4 * i, &[*s, *l, *lp, *a]);
    }
    b.put(actions, &[0x01, 0x00]); // catch type 1, end of chain
    pcrel(&mut b, type_entry as u64, TYPEINFO_SLOT);

    // --- .eh_frame ----------------------------------------------------------------
    let eh = 0x940u64;
    let mut at = eh as usize;
    let cie = at;
    b.put(at + 4, &0u32.to_le_bytes());
    b.put(at + 8, &[1]); // version
    b.put(at + 9, b"zPLR\0");
    b.put(at + 14, &[1, 0x78, 16, 7]); // code align, data align -8, return reg, aug length
    b.put(at + 18, &[0x9b]);
    pcrel(&mut b, (at + 19) as u64, PERSONALITY_SLOT);
    b.put(at + 23, &[0x1b, 0x1b]); // L, R: pcrel | sdata4
    let cie_len = 25usize.next_multiple_of(8);
    b.u32(at, (cie_len - 4) as u32);
    at += cie_len;
    let mut fdes = Vec::new();
    for (start, range, lsda_ptr) in [
        (0x700u64, 0x10u32, Some(0x800u64)),
        (0x710, 0x10, None),
        (0x600, 0x60, None),
    ] {
        let fde = at;
        b.u32(at + 4, (at + 4 - cie) as u32);
        pcrel(&mut b, (at + 8) as u64, start);
        b.u32(at + 12, range);
        b.put(at + 16, &[4]);
        match lsda_ptr {
            Some(l) => pcrel(&mut b, (at + 17) as u64, l),
            None => b.u32(at + 17, 0),
        }
        let len = 21usize.next_multiple_of(8);
        b.u32(at, (len - 4) as u32);
        at += len;
        fdes.push((start, fde as u64));
    }
    b.u32(at, 0); // terminator
    // .eh_frame_hdr: version, pcrel|sdata4 pointer, udata4 count, datarel|sdata4 table.
    b.put(0x900, &[1, 0x1b, 0x03, 0x3b]);
    pcrel(&mut b, 0x904, eh);
    b.u32(0x908, 3);
    let mut sorted = fdes.clone();
    sorted.sort();
    for (i, (start, fde)) in sorted.iter().enumerate() {
        b.u32(0x90c + 8 * i, (*start as i64 - 0x900) as i32 as u32);
        b.u32(0x910 + 8 * i, (*fde as i64 - 0x900) as i32 as u32);
    }

    // --- the second segment -------------------------------------------------------
    let dynamic: [(i64, u64); 21] = [
        (1, u64::from(libc)),
        (12, 0x730),
        (13, 0x740),
        (25, INIT_ARRAY),
        (27, 16),
        (5, 0x300),
        (6, 0x200),
        (10, 0xa0),
        (11, 24),
        (7, 0x420),
        (8, 7 * 24),
        (23, 0x500),
        (2, 48),
        (20, 7),
        (0x6fff_fff0, 0x3a0),
        (0x6fff_fffe, 0x3c0),
        (0x6fff_ffff, 1),
        (4, 0x3e0),
        (3, 0x3200),
        (0, 0),
        (0, 0),
    ];
    for (i, (t, v)) in dynamic.iter().enumerate() {
        b.u64(0x1000 + 16 * i, *t as u64);
        b.u64(0x1000 + 16 * i + 8, *v);
    }
    b.u64(0x1308, 0x718); // a non-PIE-style absolute slot: no relocation
    b.put(0x1400, &[1, 2, 3, 4, 5, 6, 7, 8]); // .tdata
    b.0.resize(0x1408, 0);

    // --- headers ----------------------------------------------------------------------
    b.put(0, b"\x7fELF");
    b.put(4, &[2, 1, 1]);
    b.u16(16, 3); // ET_DYN
    b.u16(18, 62);
    b.u32(20, 1);
    b.u64(24, 0x700);
    b.u64(32, 64);
    b.u16(52, 64);
    b.u16(54, 56);
    b.u16(56, 6);
    b.u16(58, 64);
    let phdrs: [(u32, u32, u64, u64, u64, u64, u64); 6] = [
        (1, 5, 0, 0, 0xa00, 0xa00, 0x1000),                // PT_LOAD R+X
        (1, 6, 0x1000, 0x3000, 0x408, 0x600, 0x1000),      // PT_LOAD R+W
        (2, 6, 0x1000, 0x3000, 0x150, 0x150, 8),           // PT_DYNAMIC
        (7, 4, 0x1400, 0x3400, 8, 0x20, 8),                // PT_TLS
        (0x6474_e550, 4, 0x900, 0x900, 0x24, 0x24, 4),     // PT_GNU_EH_FRAME
        (0x6474_e552, 4, 0x1000, 0x3000, 0x200, 0x200, 1), // PT_GNU_RELRO
    ];
    for (i, (ty, flags, off, va, filesz, memsz, align)) in phdrs.iter().enumerate() {
        let at = 64 + 56 * i;
        b.u32(at, *ty);
        b.u32(at + 4, *flags);
        b.u64(at + 8, *off);
        b.u64(at + 16, *va);
        b.u64(at + 24, *va);
        b.u64(at + 32, *filesz);
        b.u64(at + 40, *memsz);
        b.u64(at + 48, *align);
    }

    if sections {
        add_sections(&mut b);
    }
    b.0
}

/// `.symtab` (one local function), `.strtab`, `.shstrtab` and the section
/// headers, after the segments.
fn add_sections(b: &mut Buf) {
    let mut strtab = Names::new();
    let local = strtab.add("local_fn");
    b.put(0x1440 + 24, &[]);
    let sym = 0x1440 + 24;
    b.u32(sym, local);
    b.put(sym + 4, &[0x02]); // local FUNC
    b.u16(sym + 6, 7);
    b.u64(sym + 8, 0x710);
    b.u64(sym + 16, 0x10);
    let strtab_at = 0x1440 + 48;
    b.put(strtab_at, &strtab.bytes);

    let mut shstr = Names::new();
    // (name, type, flags, addr, offset, size, link, entsize)
    let secs: [(&str, u32, u64, u64, u64, u64, u32, u64); 17] = [
        (".dynsym", 11, 2, 0x200, 0x200, 24 * 8, 2, 24),
        (".dynstr", 3, 2, 0x300, 0x300, 0xa0, 0, 0),
        (".gnu.version", 0x6fff_ffff, 2, 0x3a0, 0x3a0, 16, 1, 2),
        (".rela.dyn", 4, 2, 0x420, 0x420, 7 * 24, 1, 24),
        (".rela.plt", 4, 2, 0x500, 0x500, 48, 1, 24),
        (".plt", 1, 6, 0x600, 0x600, 0x20, 0, 16),
        (".plt.got", 1, 6, 0x640, 0x640, 8, 0, 8),
        (".plt.sec", 1, 6, 0x650, 0x650, 16, 0, 16),
        (".text", 1, 6, 0x700, 0x700, 0x60, 0, 0),
        (".gcc_except_table", 1, 2, 0x800, 0x800, 0x20, 0, 0),
        (".eh_frame_hdr", 1, 2, 0x900, 0x900, 0x24, 0, 0),
        (".eh_frame", 1, 2, 0x940, 0x940, 0xa0, 0, 0),
        (".dynamic", 6, 3, 0x3000, 0x1000, 0x150, 2, 16),
        (".init_array", 14, 3, INIT_ARRAY, 0x1300, 16, 0, 8),
        (".tbss", 8, 0x403, 0x3410, 0x1410, 0x10, 0, 0),
        (".symtab", 2, 0, 0, 0x1440, 48, 17, 24),
        (
            ".strtab",
            3,
            0,
            0,
            strtab_at as u64,
            strtab.bytes.len() as u64,
            0,
            0,
        ),
    ];
    let names: Vec<u32> = secs.iter().map(|s| shstr.add(s.0)).collect();
    let shstr_name = shstr.add(".shstrtab");
    let shstr_at = 0x1500;
    b.put(shstr_at, &shstr.bytes);
    let shoff = 0x1600usize;
    for (i, s) in secs.iter().enumerate() {
        let at = shoff + 64 * (i + 1);
        b.u32(at, names[i]);
        b.u32(at + 4, s.1);
        b.u64(at + 8, s.2);
        b.u64(at + 16, s.3);
        b.u64(at + 24, s.4);
        b.u64(at + 32, s.5);
        b.u32(at + 40, s.6);
        b.u64(at + 56, s.7);
    }
    let last = shoff + 64 * (secs.len() + 1);
    b.u32(last, shstr_name);
    b.u32(last + 4, 3);
    b.u64(last + 24, shstr_at as u64);
    b.u64(last + 32, shstr.bytes.len() as u64);
    b.0.resize(last + 64, 0);
    b.u64(40, shoff as u64);
    b.u16(60, (secs.len() + 2) as u16);
    b.u16(62, (secs.len() + 1) as u16);
}
