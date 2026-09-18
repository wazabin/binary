//! The name -> address index behind [`BinaryFormat::symbol_address`].
//!
//! [`BinaryFormat::symbol_address`]: crate::BinaryFormat::symbol_address

use std::collections::HashMap;

/// Function names mapped to their addresses, built once at parse time so a
/// by-name lookup does not rescan the function table.
#[derive(Clone, Default)]
pub(crate) struct SymbolIndex {
    /// Each name's address, and whether that address is an external (import)
    /// stub a later definition of the same name may replace.
    by_name: HashMap<String, (u64, bool)>,
}

impl SymbolIndex {
    /// Build the index from `(name, address, is_external)` triples, visited
    /// in ascending address order.
    ///
    /// When several functions share a name, a defined function beats an
    /// import stub, and otherwise the lowest address wins, so the result is
    /// deterministic for the duplicated `static` helpers a linker keeps in
    /// `.symtab`.
    pub(crate) fn build<'a>(
        functions: impl IntoIterator<Item = (Option<&'a str>, u64, bool)>,
    ) -> Self {
        let mut index = Self::default();
        for (name, address, is_external) in functions {
            let Some(name) = name.filter(|name| !name.is_empty()) else {
                continue;
            };
            match index.by_name.get_mut(name) {
                Some(bound @ (_, true)) if !is_external => *bound = (address, false),
                Some(_) => {}
                None => {
                    index
                        .by_name
                        .insert(name.to_owned(), (address, is_external));
                }
            }
        }
        index
    }

    /// The address bound to `name`, if any.
    pub(crate) fn address(&self, name: &str) -> Option<u64> {
        self.by_name.get(name).map(|&(address, _)| address)
    }
}

impl std::fmt::Debug for SymbolIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolIndex")
            .field("names", &self.by_name.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::SymbolIndex;

    #[test]
    fn lowest_address_wins_between_definitions() {
        let index = SymbolIndex::build([
            (Some("helper"), 0x1000, false),
            (Some("helper"), 0x2000, false),
        ]);
        assert_eq!(index.address("helper"), Some(0x1000));
    }

    #[test]
    fn definition_beats_import_stub_regardless_of_order() {
        let index = SymbolIndex::build([
            (Some("memcpy"), 0x1000, true),
            (Some("memcpy"), 0x2000, false),
        ]);
        assert_eq!(index.address("memcpy"), Some(0x2000));

        let index = SymbolIndex::build([
            (Some("memcpy"), 0x1000, false),
            (Some("memcpy"), 0x2000, true),
        ]);
        assert_eq!(index.address("memcpy"), Some(0x1000));
    }

    #[test]
    fn nameless_and_empty_names_are_skipped() {
        let index = SymbolIndex::build([(None, 0x1000, false), (Some(""), 0x2000, false)]);
        assert_eq!(index.address(""), None);
    }
}
