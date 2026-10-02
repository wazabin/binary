//! The exception tables, read the way the unwinder reads them: the
//! `.eh_frame_hdr` -> the FDEs -> each FDE's LSDA (the call-site records,
//! their landing pads and action chains), the personality from the FDE's CIE.
//! [`Image::eh`] is the way in; the CIE / FDE walk itself is
//! [`eh_frame`](crate::eh_frame).
//!
//! A binary with no `.eh_frame_hdr` (or an emptied one) has no exception
//! handling here: the reader finds nothing, as the unwinder would.
//!
//! Nothing here knows a language: the clauses of a landing pad are the action
//! chain, entry by entry; a type-table entry is an address ([`Ptr`]).

use std::collections::BTreeMap;

use crate::eh_frame::{Cursor, OMIT, enc_size, fdes_from_hdr};
use crate::image::{Image, RelocKind, SymbolKind};

/// A pointer the tables name, resolved the way the loader would: an import
/// (a GOT / PLT slot with a symbolic relocation, or a PLT stub) by its name,
/// anything else by its value in the original address space.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ptr {
    Va(u64),
    Sym(String),
}

#[derive(Clone, Debug)]
pub enum Clause {
    /// `catch` of the type-table entry `index` (its value; 0 = catch-all).
    Catch {
        index: i64,
        ty: Ptr,
    },
    Cleanup,
    /// An exception specification: `index` (negative) and its types.
    Filter {
        index: i64,
        types: Vec<Ptr>,
    },
}

/// One call-site record of an LSDA, in absolute addresses.
#[derive(Clone, Debug)]
pub struct Site {
    pub start: u64,
    pub end: u64,
    /// The landing pad, when the call has one.
    pub pad: Option<u64>,
    pub clauses: Vec<Clause>,
}

/// One function that has an LSDA.
#[derive(Clone, Debug)]
pub struct FuncEh {
    pub start: u64,
    pub end: u64,
    pub personality: Option<Ptr>,
    pub sites: Vec<Site>,
    pub lsda: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Eh {
    /// Sorted by start.
    pub funcs: Vec<FuncEh>,
    /// How many FDEs the header lists.
    pub fdes: usize,
    pub warnings: Vec<String>,
}

// --- pointers -------------------------------------------------------------------

struct Resolver<'i, 'a> {
    img: &'i Image<'a>,
    relocs: BTreeMap<u64, &'i crate::image::Reloc>,
}

impl<'i, 'a> Resolver<'i, 'a> {
    fn new(img: &'i Image<'a>) -> Self {
        Resolver {
            img,
            relocs: img.relocs.iter().map(|r| (r.offset, r)).collect(),
        }
    }

    /// `value` read with `enc`: a slot to look through when indirect.
    fn ptr(&self, enc: u8, value: u64) -> Option<Ptr> {
        if enc & 0x80 != 0 {
            return self.slot(value);
        }
        Some(self.direct(value))
    }

    fn direct(&self, va: u64) -> Ptr {
        match self.img.import_name(va) {
            Some(n) => Ptr::Sym(n.to_string()),
            None => Ptr::Va(va),
        }
    }

    fn slot(&self, slot: u64) -> Option<Ptr> {
        if let Some(r) = self.relocs.get(&slot) {
            let sym = self.img.reloc_symbol(r);
            let tls = sym.is_some_and(|s| s.kind == SymbolKind::Tls);
            // A PIE's own symbol binds to its definition (`DW.ref` of a
            // typeinfo the program defines and exports).
            if let Some(s) = sym.filter(|s| self.img.is_pie() && r.sym != 0 && s.defined)
                && r.kind != RelocKind::Copy
                && !tls
            {
                return Some(self.direct(s.addr.wrapping_add(r.addend as u64)));
            }
            let name = sym.map(|s| s.name).unwrap_or("");
            match r.kind {
                RelocKind::JumpSlot | RelocKind::GlobDat | RelocKind::Abs
                    if !name.is_empty() && r.addend == 0 =>
                {
                    return Some(Ptr::Sym(name.to_string()));
                }
                RelocKind::Relative => return Some(Ptr::Va(r.addend as u64)),
                RelocKind::Abs if name.is_empty() => return Some(Ptr::Va(r.addend as u64)),
                _ => {}
            }
        }
        let w = if self.img.is_64() { 8 } else { 4 };
        let b = self.img.read(slot, w)?;
        let v = if w == 8 {
            u64::from_le_bytes(b.try_into().ok()?)
        } else {
            u64::from(u32::from_le_bytes(b.try_into().ok()?))
        };
        Some(self.direct(v))
    }
}

// --- the LSDA -------------------------------------------------------------------

/// An LSDA as encoded: everything but the function it belongs to.
struct RawLsda<'a> {
    at: u64,
    /// From `at` to the end of its segment.
    b: &'a [u8],
    /// The landing-pad base when the header gives one (else: the function).
    lp_start: Option<u64>,
    tt_enc: u8,
    tt_base: u64,
    /// Offset in `b` of the action table.
    actions: usize,
    /// `(start, length, pad, action)` as encoded: `start` and `length` are
    /// function-relative, `pad` relative to the landing-pad base, `action`
    /// 1-based into the action table (0 = none).
    cs: Vec<(u64, u64, u64, u64)>,
}

/// Parse the LSDA at `at` (`b` = the bytes from there to the end of the
/// segment).
fn parse_raw(b: &[u8], at: u64, ptr_size: usize) -> Option<RawLsda<'_>> {
    let mut c = Cursor::new(b, at);
    c.ptr_size = ptr_size;
    let lp_enc = c.u8()?;
    let lp_start = if lp_enc == OMIT {
        None
    } else {
        Some(c.encoded(lp_enc)?)
    };
    let tt_enc = c.u8()?;
    let mut tt_base = 0u64;
    if tt_enc != OMIT {
        let tt_off = c.uleb()?;
        tt_base = c.addr() + tt_off;
    }
    let cs_enc = c.u8()?;
    let cs_len = c.uleb()? as usize;
    let cs_start = c.p;
    let actions = cs_start.checked_add(cs_len)?;
    if actions > b.len() {
        return None;
    }
    let mut cs: Vec<(u64, u64, u64, u64)> = Vec::new();
    while c.p < actions && cs.len() < 65536 {
        let s = c.encoded(cs_enc & 0x0f)?;
        let l = c.encoded(cs_enc & 0x0f)?;
        let lp = c.encoded(cs_enc & 0x0f)?;
        let a = c.uleb()?;
        cs.push((s, l, lp, a));
    }
    Some(RawLsda {
        at,
        b,
        lp_start,
        tt_enc,
        tt_base,
        actions,
        cs,
    })
}

/// The call sites of `raw` as they apply to the function at `func`, their
/// pads and clauses resolved. `None` when an action chain or a type-table
/// entry is unreadable.
fn sites_of(rs: &Resolver, raw: &RawLsda, func: u64) -> Option<Vec<Site>> {
    let img = rs.img;
    let ptr_size = if img.is_64() { 8 } else { 4 };
    let lp_start = raw.lp_start.unwrap_or(func);
    let (tt_enc, tt_base) = (raw.tt_enc, raw.tt_base);
    let seg_end = raw.at + raw.b.len() as u64;
    // A type-table entry: index i >= 1 sits i entries below the base.
    let type_at = |i: i64| -> Option<Ptr> {
        if tt_enc == OMIT || i <= 0 {
            return None;
        }
        let sz = enc_size(tt_enc, ptr_size);
        if sz == 0 {
            return None;
        }
        let at = tt_base.wrapping_sub(sz as u64 * i as u64);
        let b = img.read(at, sz)?;
        let mut cc = Cursor::new(b, at);
        cc.ptr_size = ptr_size;
        let v = cc.encoded(tt_enc)?;
        if v == 0 {
            return Some(Ptr::Va(0));
        }
        rs.ptr(tt_enc, v)
    };
    let spec_at = |i: i64| -> Option<Vec<Ptr>> {
        // The list of type indices, uleb, 0-terminated, at tt_base + (-i - 1).
        let at = tt_base.wrapping_add((-i - 1) as u64);
        let left = (seg_end as usize).saturating_sub(at as usize);
        let b = img.read(at, 64.min(left))?;
        let mut cc = Cursor::new(b, at);
        let mut out = Vec::new();
        loop {
            let t = cc.uleb()?;
            if t == 0 {
                break;
            }
            out.push(type_at(t as i64)?);
        }
        Some(out)
    };
    let mut sites = Vec::new();
    for &(s, l, lp, a) in &raw.cs {
        let mut clauses = Vec::new();
        if lp != 0 {
            if a == 0 {
                clauses.push(Clause::Cleanup);
            } else {
                let mut p = raw.actions + a as usize - 1;
                for _ in 0..256 {
                    let mut ac = Cursor::new(raw.b, raw.at);
                    ac.p = p;
                    let filter = ac.sleb()?;
                    let next_at = ac.p;
                    let next = ac.sleb()?;
                    if filter > 0 {
                        clauses.push(Clause::Catch {
                            index: filter,
                            ty: type_at(filter)?,
                        });
                    } else if filter == 0 {
                        clauses.push(Clause::Cleanup);
                    } else {
                        clauses.push(Clause::Filter {
                            index: filter,
                            types: spec_at(filter)?,
                        });
                    }
                    if next == 0 {
                        break;
                    }
                    p = (next_at as i64 + next) as usize;
                }
            }
        }
        sites.push(Site {
            start: func.wrapping_add(s),
            end: func.wrapping_add(s).wrapping_add(l),
            pad: (lp != 0).then(|| lp_start.wrapping_add(lp)),
            clauses,
        });
    }
    Some(sites)
}

// --- the FDE-based reader -----------------------------------------------------------

/// Every FDE with an LSDA, its call sites' pads and clauses resolved; the
/// personality from the CIE.
pub(crate) fn parse(img: &Image) -> Eh {
    let mut eh = Eh::default();
    let Some((hdr, len)) = img.eh_frame_hdr() else {
        return eh;
    };
    let ptr_size = if img.is_64() { 8 } else { 4 };
    let mem = |va: u64| img.bytes_from(va);
    let Some((_, fdes)) = fdes_from_hdr(&mem, hdr, len, ptr_size) else {
        eh.warnings.push(format!(
            "PT_GNU_EH_FRAME at {hdr:#x}: unreadable .eh_frame_hdr; exception handling not lifted"
        ));
        return eh;
    };
    let rs = Resolver::new(img);
    eh.fdes = fdes.len();
    for f in &fdes {
        let Some(l) = f.lsda else { continue };
        let lsda = if f.lsda_indirect {
            match img.read(l, ptr_size) {
                Some(b) if ptr_size == 8 => u64::from_le_bytes(b.try_into().unwrap()),
                Some(b) => u64::from(u32::from_le_bytes(b.try_into().unwrap())),
                None => continue,
            }
        } else {
            l
        };
        if lsda == 0 {
            continue;
        }
        let personality = f.cie.personality.and_then(|(enc, v)| rs.ptr(enc, v));
        let sites = img
            .bytes_from(lsda)
            .and_then(|b| parse_raw(b, lsda, ptr_size))
            .and_then(|raw| sites_of(&rs, &raw, f.start));
        match sites {
            Some(sites) => eh.funcs.push(FuncEh {
                start: f.start,
                end: f.end,
                personality,
                sites,
                lsda,
            }),
            None => eh
                .warnings
                .push(format!("fde {:#x}: LSDA at {lsda:#x} unreadable", f.start)),
        }
    }
    eh.funcs.sort_by_key(|f| f.start);
    eh.funcs.dedup_by_key(|f| f.start);
    eh
}
