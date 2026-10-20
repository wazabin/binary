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

use std::collections::{BTreeMap, HashMap, HashSet};

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
        tt_base = c.addr().wrapping_add(tt_off);
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
///
/// Every site, clause and filter type read takes one unit of `budget` (see
/// [`take`]); when
/// it runs out the LSDA is `None` and `budget` is 0.
fn sites_of(rs: &Resolver, raw: &RawLsda, func: u64, budget: &mut usize) -> Option<Vec<Site>> {
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
        let at = tt_base.wrapping_sub((sz as u64).checked_mul(i as u64)?);
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
        // The list of type indices, uleb, 0-terminated, at tt_base + (-i - 1)
        // (`!i`, which `i64::MIN` does not overflow).
        let at = tt_base.wrapping_add(!i as u64);
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
    // The clauses of the action chain at `a` (1-based), read once per LSDA:
    // the call sites share chains. A chain ends at 0, at 256 records, or
    // where it comes back to a record it has read (a cycle the unwinder would
    // spin on).
    let mut chains: HashMap<u64, Vec<Clause>> = HashMap::new();
    let mut chain = |a: u64, budget: &mut usize| -> Option<Vec<Clause>> {
        if let Some(c) = chains.get(&a) {
            return Some(c.clone());
        }
        let mut clauses = Vec::new();
        let mut p = usize::try_from(a - 1)
            .ok()
            .and_then(|a| raw.actions.checked_add(a))?;
        let mut seen = HashSet::new();
        for _ in 0..256 {
            if !seen.insert(p) {
                break;
            }
            let mut ac = Cursor::new(raw.b, raw.at);
            ac.p = p;
            let filter = ac.sleb()?;
            let next_at = ac.p;
            let next = ac.sleb()?;
            let clause = if filter > 0 {
                Clause::Catch {
                    index: filter,
                    ty: type_at(filter)?,
                }
            } else if filter == 0 {
                Clause::Cleanup
            } else {
                Clause::Filter {
                    index: filter,
                    types: spec_at(filter)?,
                }
            };
            let cost = 1 + match &clause {
                Clause::Filter { types, .. } => types.len(),
                _ => 0,
            };
            take(budget, cost)?;
            clauses.push(clause);
            if next == 0 {
                break;
            }
            p = usize::try_from((next_at as i64).checked_add(next)?).ok()?;
        }
        chains.insert(a, clauses.clone());
        Some(clauses)
    };
    let mut sites = Vec::new();
    for &(s, l, lp, a) in &raw.cs {
        take(budget, 1)?;
        let mut clauses = Vec::new();
        if lp != 0 {
            if a == 0 {
                clauses.push(Clause::Cleanup);
            } else {
                // A cached chain costs its length again: the copy is memory too.
                clauses = chain(a, budget)?;
                take(budget, clauses.len())?;
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

/// Takes `n` units of `budget`, or empties it and is `None` when it has
/// fewer.
fn take(budget: &mut usize, n: usize) -> Option<()> {
    match budget.checked_sub(n) {
        Some(left) => {
            *budget = left;
            Some(())
        }
        None => {
            *budget = 0;
            None
        }
    }
}

/// How many sites, clauses and filter types [`parse`] reads from one image,
/// at most: an LSDA's chains are shared by its sites and an LSDA by its
/// FDEs, so a small file can name far more than it holds.
const BUDGET: usize = 1 << 22;

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
    let mut budget = BUDGET;
    // The (start, LSDA) pairs read, and the starts lifted: the header may
    // list one FDE many times, and only the first lifted read of a start is
    // kept.
    let mut read = HashSet::new();
    let mut lifted = HashSet::new();
    for f in &fdes {
        let Some(l) = f.lsda else { continue };
        if lifted.contains(&f.start) {
            continue;
        }
        let lsda = if f.lsda_indirect {
            match img.read(l, ptr_size) {
                Some(b) if ptr_size == 8 => u64::from_le_bytes(b.try_into().unwrap()),
                Some(b) => u64::from(u32::from_le_bytes(b.try_into().unwrap())),
                None => continue,
            }
        } else {
            l
        };
        if lsda == 0 || !read.insert((f.start, lsda)) {
            continue;
        }
        let personality = f.cie.personality.and_then(|(enc, v)| rs.ptr(enc, v));
        let sites = img
            .bytes_from(lsda)
            .and_then(|b| parse_raw(b, lsda, ptr_size))
            .and_then(|raw| sites_of(&rs, &raw, f.start, &mut budget));
        match sites {
            Some(sites) => {
                lifted.insert(f.start);
                eh.funcs.push(FuncEh {
                    start: f.start,
                    end: f.end,
                    personality,
                    sites,
                    lsda,
                })
            }
            None if budget == 0 => {
                eh.warnings.push(format!(
                    "fde {:#x}: the exception tables name more than {BUDGET} call sites and clauses; \
                     the rest not lifted",
                    f.start
                ));
                break;
            }
            None => eh
                .warnings
                .push(format!("fde {:#x}: LSDA at {lsda:#x} unreadable", f.start)),
        }
    }
    eh.funcs.sort_by_key(|f| f.start);
    eh.funcs.dedup_by_key(|f| f.start);
    eh
}
