//! The [`TargetOs`] enum: a loaded binary's operating system.
//!
//! Lives here (not in `qcode` core) so `wazabin_binary`'s [`BinaryFormat::os`] can name
//! it while `wazabin_binary` stays a leaf below `qcode`. `qcode` core re-exports it as
//! `qcode::context::TargetOs`, so existing paths keep compiling.
//!
//! [`BinaryFormat::os`]: crate::BinaryFormat::os

/// The operating system of a loaded binary, inferred from its container format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetOs {
    #[default]
    Unknown,
    Windows,
    Linux,
}
