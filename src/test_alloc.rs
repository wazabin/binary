//! A test allocator that refuses any single allocation over [`CAP`], so a
//! reader that sizes a buffer from an untrusted header value fails the test
//! (the refusal aborts the test binary with "memory allocation of N bytes
//! failed") instead of paging in gigabytes. [`CAP`] sits above
//! [`ZERO_FILL_LIMIT`](crate::ZERO_FILL_LIMIT) with room for a region's
//! file bytes.

use std::alloc::{GlobalAlloc, Layout, System};

/// 320 MiB.
pub const CAP: usize = 320 << 20;

struct Capped;

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;
