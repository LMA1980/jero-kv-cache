//! Core context memory holder state and shared memory arena management.
//!
//! Provides thread-safe arena allocation, mapping registry, deallocation,
//! and strict operator `LimitMEMLOCK` ceiling enforcement (28 GiB reference
//! board ceiling, ADR-037 / AGENTS.md Rule 5).

use core::mem::{align_of, size_of};
use std::collections::HashMap;
use std::fmt;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::sync::{Arc, RwLock};

use zerocopy::{FromBytes, IntoBytes};

use jero_core::K3Error;
use jero_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS, CACHELINE_ALIGNMENT, KvBlockMeta,
    OutputSlabHeader, RecurrentStateSlotHeader, SLAB_STATUS_IDLE, TAG_FREE, TokenInputSlotHeader,
};
use jero_core::ipc::protocol::{
    AllocateArenaRequest, KV_DTYPE_F16, KV_DTYPE_F32, KV_DTYPE_Q8_0, StatusCode,
};
use jero_core::ipc::shm::{
    ArenaOptions, DEFAULT_LIMIT_MEMLOCK_BYTES, IpcError, SharedMemoryArena, system_page_size,
};
use jero_core::{KvCacheView, KvGeometry};

/// Per-layer KV cache retention policy for hybrid local/global attention models (ADR-027, ADR-037).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKvPolicy {
    /// Full-context global attention retaining up to `n_ctx` tokens.
    GlobalFullContext,
    /// Sliding-window local attention retaining at most `window_tokens` tokens.
    SlidingWindow {
        /// Maximum sliding window token span.
        window_tokens: u32,
    },
}

/// Computes total 16-token KV layer-blocks across all layers for a hybrid local/global schedule.
#[must_use]
pub fn compute_hybrid_kv_block_count(n_ctx: u32, layer_policies: &[LayerKvPolicy]) -> u32 {
    if n_ctx == 0 || layer_policies.is_empty() {
        return 0;
    }

    let mut total_blocks = 0_u32;
    for &policy in layer_policies {
        let layer_tokens = match policy {
            LayerKvPolicy::GlobalFullContext => n_ctx,
            LayerKvPolicy::SlidingWindow { window_tokens } => n_ctx.min(window_tokens),
        };
        let layer_blocks = if layer_tokens == 0 {
            0
        } else {
            layer_tokens.div_ceil(BLOCK_SIZE_TOKENS)
        };
        total_blocks = total_blocks.saturating_add(layer_blocks);
    }
    total_blocks
}

/// Sizing and geometry parameters for allocating a shared memory arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaAllocationConfig {
    /// Logical model identifier or name.
    pub model_id: String,
    /// Maximum context tokens (`n_ctx`).
    pub n_ctx: u32,
    /// Number of active slots (typically 1 for single-stream worker).
    pub active_slots: u32,
    /// Attention head dimension (e.g. 64, 128).
    pub head_dim: u32,
    /// Number of transformer layers in the model.
    pub num_layers: u32,
    /// Number of key-value attention heads per layer (GQA/MQA models).
    pub head_count_kv: u32,
    /// Model vocabulary size (dimension of output logits).
    pub vocab_size: u32,
    /// KV cache element encoding (`0 = F32`, `1 = F16`, `2 = Q8_0`).
    pub kv_dtype: u8,
    /// Whether to mlock the mapped memory pages into physical RAM.
    pub lock_memory: bool,
    /// Explicit capacity override in bytes (e.g. from `AllocateArenaRequest`).
    pub explicit_bytes: Option<u64>,
    /// Explicit block count override (e.g. `max_blocks` from `AllocateArenaRequest`).
    pub max_blocks: Option<u32>,
    /// Optional per-layer hybrid local/global KV retention schedule.
    pub layer_policies: Option<Vec<LayerKvPolicy>>,
    /// Optional compressed `MLA` per-token KV latent stride (`kv_lora_rank + qk_rope_head_dim`).
    pub kv_latent_dim: Option<u32>,
    /// Number of linear-time recurrent/SSM layers persisted via ping-pong banks.
    pub num_recurrent_layers: u32,
    /// Per-layer convolution history byte requirement in each recurrent bank.
    pub conv_state_bytes_per_layer: u32,
    /// Per-layer SSM/WKV recurrent state byte requirement in each recurrent bank.
    pub ssm_state_bytes_per_layer: u32,
}

impl ArenaAllocationConfig {
    /// Creates a new configuration with standard defaults for the given model dimensions.
    #[must_use]
    pub fn new(model_id: impl Into<String>, n_ctx: u32, head_dim: u32, num_layers: u32) -> Self {
        Self {
            model_id: model_id.into(),
            n_ctx,
            active_slots: 1,
            head_dim,
            num_layers,
            head_count_kv: 2,
            vocab_size: 32000,
            kv_dtype: KV_DTYPE_F32,
            lock_memory: false,
            explicit_bytes: None,
            max_blocks: None,
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers: 0,
            conv_state_bytes_per_layer: 0,
            ssm_state_bytes_per_layer: 0,
        }
    }

    /// Creates a pure recurrent/SSM arena configuration (`O(1)` in `n_ctx`) with zero KV layers.
    #[must_use]
    pub fn new_recurrent(
        model_id: impl Into<String>,
        n_ctx: u32,
        vocab_size: u32,
        num_recurrent_layers: u32,
        conv_state_bytes_per_layer: u32,
        ssm_state_bytes_per_layer: u32,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            n_ctx,
            active_slots: 1,
            head_dim: 0,
            num_layers: 0,
            head_count_kv: 0,
            vocab_size,
            kv_dtype: KV_DTYPE_F32,
            lock_memory: false,
            explicit_bytes: None,
            max_blocks: None,
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers,
            conv_state_bytes_per_layer,
            ssm_state_bytes_per_layer,
        }
    }

    /// Derives validated configuration from an incoming [`AllocateArenaRequest`] message.
    ///
    /// Fails closed with [`KvError::InvalidDimensions`] when any capacity or geometry
    /// dimension is zero, `kv_dtype` is unsupported, or dimensions exceed platform `usize`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `req.validate()` fails or any dimension
    /// overflows `usize`.
    pub fn from_request(req: &AllocateArenaRequest) -> Result<Self, KvError> {
        req.validate()
            .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
        usize::try_from(req.n_ctx)
            .map_err(|_| KvError::InvalidDimensions("n_ctx overflowed usize".to_owned()))?;
        usize::try_from(req.max_blocks)
            .map_err(|_| KvError::InvalidDimensions("max_blocks overflowed usize".to_owned()))?;
        usize::try_from(req.num_layers)
            .map_err(|_| KvError::InvalidDimensions("num_layers overflowed usize".to_owned()))?;
        usize::try_from(req.head_count_kv)
            .map_err(|_| KvError::InvalidDimensions("head_count_kv overflowed usize".to_owned()))?;
        usize::try_from(req.head_dim)
            .map_err(|_| KvError::InvalidDimensions("head_dim overflowed usize".to_owned()))?;
        usize::try_from(req.vocab_size)
            .map_err(|_| KvError::InvalidDimensions("vocab_size overflowed usize".to_owned()))?;

        Ok(Self {
            model_id: req.model_id.clone(),
            n_ctx: req.n_ctx,
            active_slots: 1,
            head_dim: req.head_dim,
            num_layers: req.num_layers,
            head_count_kv: req.head_count_kv,
            vocab_size: req.vocab_size,
            kv_dtype: req.kv_dtype,
            lock_memory: false,
            explicit_bytes: Some(req.arena_bytes),
            max_blocks: Some(req.max_blocks),
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers: 0,
            conv_state_bytes_per_layer: 0,
            ssm_state_bytes_per_layer: 0,
        })
    }

    /// Sets the number of active slots.
    #[must_use]
    pub const fn with_active_slots(mut self, active_slots: u32) -> Self {
        self.active_slots = active_slots;
        self
    }

    /// Sets the key-value head count for GQA/MQA.
    #[must_use]
    pub const fn with_head_count_kv(mut self, head_count_kv: u32) -> Self {
        self.head_count_kv = head_count_kv;
        self
    }

    /// Sets the model vocabulary size for output logits.
    #[must_use]
    pub const fn with_vocab_size(mut self, vocab_size: u32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Sets whether pages should be locked in RAM via `mlock`.
    #[must_use]
    pub const fn with_lock_memory(mut self, lock: bool) -> Self {
        self.lock_memory = lock;
        self
    }

    /// Sets an explicit capacity in bytes.
    #[must_use]
    pub const fn with_explicit_bytes(mut self, bytes: u64) -> Self {
        self.explicit_bytes = Some(bytes);
        self
    }

    /// Sets an explicit maximum block count.
    #[must_use]
    pub const fn with_max_blocks(mut self, max_blocks: u32) -> Self {
        self.max_blocks = Some(max_blocks);
        self
    }

    /// Sets per-layer hybrid local/global KV retention policies.
    #[must_use]
    pub fn with_layer_policies(mut self, policies: Vec<LayerKvPolicy>) -> Self {
        self.layer_policies = Some(policies);
        self
    }

    /// Configures compressed Multi-Head Latent Attention (`MLA`) per-token KV latent width (`kv_lora_rank + qk_rope_head_dim`).
    #[must_use]
    pub const fn with_kv_latent_dim(mut self, kv_lora_rank: u32, qk_rope_head_dim: u32) -> Self {
        self.kv_latent_dim = Some(kv_lora_rank.saturating_add(qk_rope_head_dim));
        self
    }

    /// Configures ping-pong recurrent/SSM state allocation on this arena (for pure SSM or hybrid models).
    #[must_use]
    pub const fn with_recurrent_state(
        mut self,
        num_recurrent_layers: u32,
        conv_state_bytes_per_layer: u32,
        ssm_state_bytes_per_layer: u32,
    ) -> Self {
        self.num_recurrent_layers = num_recurrent_layers;
        self.conv_state_bytes_per_layer = conv_state_bytes_per_layer;
        self.ssm_state_bytes_per_layer = ssm_state_bytes_per_layer;
        self
    }

    /// Sets the KV cache element encoding (`KV_DTYPE_F32`, `KV_DTYPE_F16`, or `KV_DTYPE_Q8_0`).
    #[must_use]
    pub const fn with_kv_dtype(mut self, kv_dtype: u8) -> Self {
        self.kv_dtype = kv_dtype;
        self
    }
}

/// Detailed layout offsets and capacity computed for an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputedArenaLayout {
    /// Total capacity of the mapped shared memory arena in bytes (page aligned).
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks allocated.
    pub block_count: u32,
    /// Exact per-slot sequence capacity (`blocks_per_slot * 16`, or `0` when `num_layers == 0`).
    pub seq_capacity: u32,
    /// Byte offset to the [`KvBlockMeta`] table.
    pub meta_table_offset: u64,
    /// Byte offset to the token input slots region.
    pub token_slots_offset: u64,
    /// Byte offset to the KV cache block vector data region.
    pub kv_blocks_offset: u64,
    /// Byte offset to the output logit slabs region.
    pub output_slabs_offset: u64,
    /// Byte offset to the recurrent state region (after output slabs, or `0` when unused).
    pub recurrent_state_offset: u64,
    /// Byte stride of each ping-pong recurrent bank (`0` when unused).
    pub recurrent_bank_stride_bytes: u64,
    /// Total bytes reserved for the recurrent state header + two ping-pong banks (`0` when unused).
    pub recurrent_total_bytes: u64,
}

/// Aligns a 64-bit unsigned integer up to the nearest multiple of `align`.
const fn align_up_u64(val: u64, align: u64) -> u64 {
    let rem = val % align;
    if rem == 0 { val } else { val + (align - rem) }
}

/// Aligns a `usize` value up to the nearest multiple of `align`.
const fn align_up_usize(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 { val } else { val + (align - rem) }
}

fn compute_per_layer_plane_row_bytes(kv_dim: u64, kv_dtype: u8) -> Result<u64, KvError> {
    match kv_dtype {
        KV_DTYPE_F32 => kv_dim
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| KvError::InvalidDimensions("kv_row_bytes overflowed u64".to_owned())),
        KV_DTYPE_F16 => kv_dim
            .checked_mul(2)
            .ok_or_else(|| KvError::InvalidDimensions("kv_row_bytes overflowed u64".to_owned())),
        KV_DTYPE_Q8_0 => kv_dim
            .div_ceil(32)
            .checked_mul(34)
            .ok_or_else(|| KvError::InvalidDimensions("kv_row_bytes overflowed u64".to_owned())),
        other => Err(KvError::InvalidDimensions(format!(
            "unsupported kv_dtype {other}; expected 0 (F32), 1 (F16), or 2 (Q8_0)"
        ))),
    }
}

/// Returns the token slot capacity bounded to [`BLOCK_SIZE_TOKENS`] for pure-recurrent (`O(1)`) arenas.
const fn effective_token_slot_capacity(config: &ArenaAllocationConfig) -> u32 {
    if config.num_layers == 0 && config.num_recurrent_layers > 0 {
        BLOCK_SIZE_TOKENS
    } else {
        config.n_ctx
    }
}

/// Computes the sub-region layout offsets and page-aligned total byte capacity
/// for the specified arena configuration.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if any sizing parameter is zero or arithmetic overflows.
#[allow(clippy::too_many_lines)]
pub fn compute_arena_layout(
    config: &ArenaAllocationConfig,
) -> Result<ComputedArenaLayout, KvError> {
    if config.n_ctx == 0 {
        return Err(KvError::InvalidDimensions(
            "n_ctx must be greater than zero".to_owned(),
        ));
    }
    if config.num_layers == 0 && config.num_recurrent_layers == 0 {
        return Err(KvError::InvalidDimensions(
            "num_layers or num_recurrent_layers must be greater than zero".to_owned(),
        ));
    }
    if config.num_layers > 0 && config.head_dim == 0 && config.kv_latent_dim.is_none() {
        return Err(KvError::InvalidDimensions(
            "head_dim must be greater than zero".to_owned(),
        ));
    }
    if config.kv_latent_dim == Some(0) {
        return Err(KvError::InvalidDimensions(
            "kv_latent_dim must be greater than zero".to_owned(),
        ));
    }
    if config.num_recurrent_layers > 0
        && config.conv_state_bytes_per_layer == 0
        && config.ssm_state_bytes_per_layer == 0
    {
        return Err(KvError::InvalidDimensions(
            "recurrent state bytes per layer must be greater than zero".to_owned(),
        ));
    }

    if config.kv_dtype > KV_DTYPE_Q8_0 {
        return Err(KvError::InvalidDimensions(format!(
            "kv_dtype {} is invalid; expected 0 (F32), 1 (F16), or 2 (Q8_0)",
            config.kv_dtype
        )));
    }
    if config.num_layers > 0 && config.kv_latent_dim.is_none() {
        if config.head_count_kv == 0 {
            return Err(KvError::InvalidDimensions(
                "head_count_kv must be greater than zero".to_owned(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(KvError::InvalidDimensions(
                "vocab_size must be greater than zero".to_owned(),
            ));
        }
    }

    let active_slots = config.active_slots.max(1);
    let head_count_kv = config.head_count_kv.max(1);
    let vocab_size = config.vocab_size.max(1);

    // 1. Block count & per-slot seq_capacity: 16 tokens per block (0 for pure-recurrent O(1) arenas)
    let mut blocks_per_slot = if config.num_layers == 0 {
        0
    } else {
        config.n_ctx.div_ceil(BLOCK_SIZE_TOKENS)
    };
    if let Some(mb) = config.max_blocks {
        blocks_per_slot = blocks_per_slot.max(mb);
    }
    let block_count = blocks_per_slot
        .checked_mul(active_slots)
        .ok_or_else(|| KvError::InvalidDimensions("block_count overflowed u32".to_owned()))?;
    let seq_capacity = if config.num_layers == 0 {
        0
    } else {
        blocks_per_slot
            .checked_mul(BLOCK_SIZE_TOKENS)
            .ok_or_else(|| KvError::InvalidDimensions("seq_capacity overflowed u32".to_owned()))?
    };

    // 2. Offsets:
    // ArenaHeader: 128 bytes (0..128)
    let meta_table_offset: u64 = 128;
    let meta_size = u64::from(block_count)
        .checked_mul(size_of::<KvBlockMeta>() as u64)
        .ok_or_else(|| KvError::InvalidDimensions("meta_table_size overflowed u64".to_owned()))?;
    let meta_end = meta_table_offset
        .checked_add(meta_size)
        .ok_or_else(|| KvError::InvalidDimensions("meta_end overflowed u64".to_owned()))?;

    // TokenInputSlot: cacheline aligned (64 bytes)
    let token_slots_offset = align_up_u64(meta_end, CACHELINE_ALIGNMENT as u64);
    let slot_tokens = effective_token_slot_capacity(config);
    let per_slot_bytes = u64::from(slot_tokens)
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|t| t.checked_add(size_of::<TokenInputSlotHeader>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("token slot size overflowed u64".to_owned()))?;
    let aligned_slot_bytes = align_up_u64(per_slot_bytes, CACHELINE_ALIGNMENT as u64);
    let total_slots_bytes = aligned_slot_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| {
            KvError::InvalidDimensions("total token slots size overflowed u64".to_owned())
        })?;
    let token_slots_end = token_slots_offset
        .checked_add(total_slots_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("token_slots_end overflowed u64".to_owned()))?;

    // Contiguous layer-major KV slab `[slot][layer][K|V][pos][kv_dim]`: cacheline aligned (64 bytes)
    let kv_blocks_offset = align_up_u64(token_slots_end, CACHELINE_ALIGNMENT as u64);
    let total_kv_blocks_bytes = if config.num_layers == 0 {
        0
    } else {
        let kv_row_bytes = if let Some(latent_dim) = config.kv_latent_dim {
            u64::from(latent_dim)
                .checked_mul(size_of::<f32>() as u64)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("kv_row_bytes overflowed u64".to_owned())
                })?
        } else {
            let kv_dim = u64::from(head_count_kv)
                .checked_mul(u64::from(config.head_dim))
                .ok_or_else(|| KvError::InvalidDimensions("kv_dim overflowed u64".to_owned()))?;
            compute_per_layer_plane_row_bytes(kv_dim, config.kv_dtype)?
        };
        let planes_per_layer: u64 = if config.kv_latent_dim.is_some() { 1 } else { 2 };
        if let Some(policies) = config.layer_policies.as_deref() {
            let expected_layers = usize::try_from(config.num_layers).unwrap_or(0);
            if policies.len() != expected_layers {
                return Err(KvError::InvalidDimensions(format!(
                    "layer_policies length ({}) does not match num_layers ({expected_layers})",
                    policies.len()
                )));
            }
            let per_layer_bytes_per_token =
                planes_per_layer.checked_mul(kv_row_bytes).ok_or_else(|| {
                    KvError::InvalidDimensions(
                        "per_layer_bytes_per_token overflowed u64".to_owned(),
                    )
                })?;
            let layer_block_bytes = u64::from(BLOCK_SIZE_TOKENS)
                .checked_mul(per_layer_bytes_per_token)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("layer_block_bytes overflowed u64".to_owned())
                })?;
            let hybrid_blocks_per_slot = compute_hybrid_kv_block_count(config.n_ctx, policies);
            let total_layer_blocks = u64::from(hybrid_blocks_per_slot)
                .checked_mul(u64::from(active_slots))
                .ok_or_else(|| {
                    KvError::InvalidDimensions(
                        "hybrid total_layer_blocks overflowed u64".to_owned(),
                    )
                })?;
            total_layer_blocks
                .checked_mul(layer_block_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions(
                        "hybrid total_kv_blocks_bytes overflowed u64".to_owned(),
                    )
                })?
        } else {
            let plane_bytes = u64::from(seq_capacity)
                .checked_mul(kv_row_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("kv_plane_bytes overflowed u64".to_owned())
                })?;
            let layer_bytes = planes_per_layer.checked_mul(plane_bytes).ok_or_else(|| {
                KvError::InvalidDimensions("kv_layer_bytes overflowed u64".to_owned())
            })?;
            let slot_bytes = u64::from(config.num_layers)
                .checked_mul(layer_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("kv_slot_bytes overflowed u64".to_owned())
                })?;
            u64::from(active_slots)
                .checked_mul(slot_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("total_kv_blocks_bytes overflowed u64".to_owned())
                })?
        }
    };
    let kv_blocks_end = kv_blocks_offset
        .checked_add(total_kv_blocks_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("kv_blocks_end overflowed u64".to_owned()))?;

    // OutputSlab: cacheline aligned (64 bytes)
    let output_slabs_offset = align_up_u64(kv_blocks_end, CACHELINE_ALIGNMENT as u64);
    let per_slab_bytes = u64::from(vocab_size)
        .checked_mul(size_of::<f32>() as u64)
        .and_then(|v| v.checked_add(size_of::<OutputSlabHeader>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("slab_bytes overflowed u64".to_owned()))?;
    let aligned_slab_bytes = align_up_u64(per_slab_bytes, CACHELINE_ALIGNMENT as u64);
    let total_slabs_bytes = aligned_slab_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| KvError::InvalidDimensions("total_slabs_bytes overflowed u64".to_owned()))?;
    let output_slabs_end = output_slabs_offset
        .checked_add(total_slabs_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("output_slabs_end overflowed u64".to_owned()))?;

    // RecurrentStateSlotHeader + dual ping-pong banks immediately after output slabs (Option A)
    let (recurrent_state_offset, recurrent_bank_stride_bytes, recurrent_total_bytes) =
        if config.num_recurrent_layers > 0 {
            let align = CACHELINE_ALIGNMENT as u64;
            let conv_aligned = align_up_u64(u64::from(config.conv_state_bytes_per_layer), align);
            let ssm_aligned = align_up_u64(u64::from(config.ssm_state_bytes_per_layer), align);
            let per_layer_stride = conv_aligned.checked_add(ssm_aligned).ok_or_else(|| {
                KvError::InvalidDimensions("recurrent per-layer stride overflowed u64".to_owned())
            })?;
            let bank_stride = u64::from(config.num_recurrent_layers)
                .checked_mul(per_layer_stride)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("recurrent bank_stride overflowed u64".to_owned())
                })?;
            let offset = align_up_u64(output_slabs_end, align);
            let two_banks = 2_u64.checked_mul(bank_stride).ok_or_else(|| {
                KvError::InvalidDimensions("recurrent dual-bank size overflowed u64".to_owned())
            })?;
            let total_region = (size_of::<RecurrentStateSlotHeader>() as u64)
                .checked_add(two_banks)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("recurrent_total_bytes overflowed u64".to_owned())
                })?;
            (offset, bank_stride, total_region)
        } else {
            (0, 0, 0)
        };

    let min_required_bytes = if config.num_recurrent_layers > 0 {
        recurrent_state_offset
            .checked_add(recurrent_total_bytes)
            .ok_or_else(|| {
                KvError::InvalidDimensions("arena min_required_bytes overflowed u64".to_owned())
            })?
    } else {
        output_slabs_end
    };
    let raw_bytes = if let Some(explicit) = config.explicit_bytes {
        explicit.max(min_required_bytes)
    } else {
        min_required_bytes
    };

    let page_size = system_page_size() as u64;
    let total_bytes = align_up_u64(raw_bytes, page_size);

    Ok(ComputedArenaLayout {
        total_bytes,
        block_count,
        seq_capacity,
        meta_table_offset,
        token_slots_offset,
        kv_blocks_offset,
        output_slabs_offset,
        recurrent_state_offset,
        recurrent_bank_stride_bytes,
        recurrent_total_bytes,
    })
}

/// Metadata returned upon successful arena allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaAllocationResult {
    /// Unique identifier assigned to the allocated arena.
    pub arena_id: u64,
    /// Total capacity of the allocated arena in bytes (page aligned).
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks initialized.
    pub block_count: u32,
    /// Model identifier associated with the arena.
    pub model_id: String,
}

/// Structured error type for KV holder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// Memory allocation would exceed the operator `LimitMEMLOCK` ceiling.
    InsufficientLockableMemory {
        /// Requested byte capacity that could not be satisfied.
        required_bytes: u64,
        /// Current total allocated/locked bytes across active arenas.
        allocated_bytes: u64,
        /// Configured bounded memlock ceiling in bytes.
        memlock_ceiling_bytes: u64,
    },
    /// Requested arena ID was not found in the registry.
    ArenaNotFound(u64),
    /// Invalid geometry or configuration parameter.
    InvalidDimensions(String),
    /// Synchronization lock acquisition failure.
    LockError(String),
    /// Underlying IPC / OS operation failed (e.g. mmap, mlock).
    IpcError(String),
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientLockableMemory {
                required_bytes,
                allocated_bytes,
                memlock_ceiling_bytes,
            } => {
                write!(
                    f,
                    "[insufficient_lockable_memory] required {required_bytes} bytes exceeds remaining budget (allocated: {allocated_bytes}, ceiling: {memlock_ceiling_bytes})"
                )
            }
            Self::ArenaNotFound(id) => write!(f, "[arena_not_found] arena ID {id} not found"),
            Self::InvalidDimensions(msg) => write!(f, "[invalid_dimensions] {msg}"),
            Self::LockError(msg) => write!(f, "[lock_error] {msg}"),
            Self::IpcError(msg) => write!(f, "[ipc_error] {msg}"),
        }
    }
}

impl std::error::Error for KvError {}

impl KvError {
    /// Maps this error into its corresponding K3 protocol [`StatusCode`].
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::InsufficientLockableMemory { .. } => StatusCode::InsufficientLockableMemory,
            Self::ArenaNotFound(_) => StatusCode::ArenaNotFound,
            Self::InvalidDimensions(_) => StatusCode::CapabilityMismatch,
            Self::LockError(_) | Self::IpcError(_) => StatusCode::InternalError,
        }
    }

    /// Returns the associated process exit code.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        self.status_code().as_exit_code()
    }
}

/// Synchronized internal registry and accounting state.
#[derive(Debug)]
struct HolderInner {
    /// Active shared memory arenas indexed by unique arena ID.
    arenas: HashMap<u64, Arc<SharedMemoryArena>>,
    /// Current total allocated bytes across all registered arenas.
    allocated_bytes: u64,
}

/// Core context memory holder state managing active shared memory arenas.
///
/// Designed per ADR-042 (policy-free memory holder) and ADR-037 / AGENTS.md Rule 5
/// (bounded `LimitMEMLOCK` ceiling enforcement).
#[derive(Debug)]
pub struct KvHolder {
    /// Bounded operator `LimitMEMLOCK` ceiling in bytes (default 28 GiB).
    memlock_ceiling_bytes: u64,
    /// Monotonically incrementing arena ID generator.
    next_arena_id: AtomicU64,
    /// Thread-safe registry and accounting state.
    inner: RwLock<HolderInner>,
}

impl Default for KvHolder {
    fn default() -> Self {
        Self::new()
    }
}

impl KvHolder {
    /// Creates a new `KvHolder` with the default operator `LimitMEMLOCK` ceiling (28 GiB).
    #[must_use]
    pub fn new() -> Self {
        Self::with_memlock_ceiling(DEFAULT_LIMIT_MEMLOCK_BYTES)
    }

    /// Creates a new `KvHolder` with an explicit bounded `LimitMEMLOCK` ceiling in bytes.
    #[must_use]
    pub fn with_memlock_ceiling(ceiling_bytes: u64) -> Self {
        Self {
            memlock_ceiling_bytes: ceiling_bytes,
            next_arena_id: AtomicU64::new(1),
            inner: RwLock::new(HolderInner {
                arenas: HashMap::new(),
                allocated_bytes: 0,
            }),
        }
    }

    /// Returns the configured `LimitMEMLOCK` ceiling in bytes.
    #[must_use]
    pub const fn memlock_ceiling_bytes(&self) -> u64 {
        self.memlock_ceiling_bytes
    }

    /// Returns total bytes currently allocated and tracked across all active arenas.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> u64 {
        self.inner.read().map_or(0, |g| g.allocated_bytes)
    }

    /// Returns remaining lockable memory budget in bytes before reaching ceiling.
    #[must_use]
    pub fn available_lockable_bytes(&self) -> u64 {
        self.memlock_ceiling_bytes
            .saturating_sub(self.total_allocated_bytes())
    }

    /// Returns the count of currently registered active arenas.
    #[must_use]
    pub fn arena_count(&self) -> usize {
        self.inner.read().map_or(0, |g| g.arenas.len())
    }

    /// Returns `true` if an arena with the given ID exists.
    #[must_use]
    pub fn contains_arena(&self, arena_id: u64) -> bool {
        self.inner
            .read()
            .is_ok_and(|g| g.arenas.contains_key(&arena_id))
    }

    /// Returns a list of all currently active arena IDs.
    #[must_use]
    pub fn list_arena_ids(&self) -> Vec<u64> {
        self.inner
            .read()
            .map(|g| g.arenas.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Allocates and initializes a new per-model shared memory arena.
    ///
    /// Sizing is computed from `n_ctx`, `active_slots`, `head_dim`, and `num_layers`.
    /// Sizing is strictly accounted against the operator `LimitMEMLOCK` ceiling.
    ///
    /// # Errors
    ///
    /// - Returns [`KvError::InsufficientLockableMemory`] (exit code 11) if the requested
    ///   capacity would exceed the operator `LimitMEMLOCK` ceiling.
    /// - Returns [`KvError::InvalidDimensions`] if configuration geometry is invalid.
    /// - Returns [`KvError::IpcError`] if OS shared memory allocation or mapping fails.
    pub fn allocate(
        &self,
        config: &ArenaAllocationConfig,
    ) -> Result<ArenaAllocationResult, KvError> {
        let layout = compute_arena_layout(config)?;
        let arena_len_usize = usize::try_from(layout.total_bytes).map_err(|_| {
            KvError::InvalidDimensions("calculated arena size exceeds pointer width".to_owned())
        })?;

        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed);
        let shm_name = format!("jero_kv_{}_{}", config.model_id, arena_id);

        // Pre-flight check and atomic reservation under write lock
        {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| KvError::LockError(e.to_string()))?;
            let current_allocated = guard.allocated_bytes;
            let projected = current_allocated.checked_add(layout.total_bytes).ok_or(
                KvError::InsufficientLockableMemory {
                    required_bytes: layout.total_bytes,
                    allocated_bytes: current_allocated,
                    memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                },
            )?;

            if projected > self.memlock_ceiling_bytes {
                return Err(KvError::InsufficientLockableMemory {
                    required_bytes: layout.total_bytes,
                    allocated_bytes: current_allocated,
                    memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                });
            }

            // Tentatively reserve bytes
            guard.allocated_bytes = projected;
        }

        // Allocate SharedMemoryArena
        let opts = ArenaOptions::new(&shm_name, arena_len_usize)
            .with_lock(config.lock_memory)
            .with_memlock_ceiling(self.memlock_ceiling_bytes);

        let mut arena = match SharedMemoryArena::create_with_options(opts) {
            Ok(arena) => arena,
            Err(err) => {
                // Rollback reservation on failure
                if let Ok(mut guard) = self.inner.write() {
                    guard.allocated_bytes =
                        guard.allocated_bytes.saturating_sub(layout.total_bytes);
                }
                return match err {
                    IpcError::InsufficientLockableMemory(err) => {
                        let req = match err {
                            K3Error::InsufficientLockableMemory { required_bytes, .. } => {
                                required_bytes
                            }
                            _ => layout.total_bytes,
                        };
                        Err(KvError::InsufficientLockableMemory {
                            required_bytes: req,
                            allocated_bytes: self.total_allocated_bytes(),
                            memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                        })
                    }
                    other => Err(KvError::IpcError(other.to_string())),
                };
            }
        };

        // Initialize ArenaHeader, KvBlockMeta, and slot/slab headers
        if let Err(init_err) = initialize_arena_contents(&mut arena, config, &layout) {
            // Rollback reservation
            if let Ok(mut guard) = self.inner.write() {
                guard.allocated_bytes = guard.allocated_bytes.saturating_sub(layout.total_bytes);
            }
            return Err(init_err);
        }

        let actual_bytes = arena.len() as u64;
        let arc_arena = Arc::new(arena);

        // Commit arena to registry under write lock
        {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| KvError::LockError(e.to_string()))?;
            // Adjust reservation if actual_bytes differs from layout.total_bytes (e.g. system page size rounding)
            if actual_bytes != layout.total_bytes {
                guard.allocated_bytes = guard
                    .allocated_bytes
                    .saturating_sub(layout.total_bytes)
                    .saturating_add(actual_bytes);
            }
            guard.arenas.insert(arena_id, arc_arena);
        }

        Ok(ArenaAllocationResult {
            arena_id,
            total_bytes: actual_bytes,
            block_count: layout.block_count,
            model_id: config.model_id.clone(),
        })
    }

    /// Convenience helper allocating an arena directly from high-level request parameters.
    ///
    /// # Errors
    /// Returns [`KvError`] on failure.
    /// Allocates an arena directly from a validated [`AllocateArenaRequest`] payload.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] when geometry validation fails, or
    /// [`KvError::InsufficientLockableMemory`] when the memory ceiling is exceeded.
    pub fn allocate_from_request(
        &self,
        req: &AllocateArenaRequest,
    ) -> Result<ArenaAllocationResult, KvError> {
        let config = ArenaAllocationConfig::from_request(req)?;
        self.allocate(&config)
    }

    /// Allocates an arena from explicit model geometry and sizing parameters.
    ///
    /// Fails closed via [`AllocateArenaRequest::validate`] when any dimension or capacity is
    /// zero or when `kv_dtype` is unsupported.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if validation fails or [`KvError`] if allocation
    /// exceeds the configured `LimitMEMLOCK` ceiling.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_arena(
        &self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
        num_layers: u32,
        head_count_kv: u32,
        head_dim: u32,
        vocab_size: u32,
        kv_dtype: u8,
    ) -> Result<ArenaAllocationResult, KvError> {
        let req = AllocateArenaRequest {
            model_id: model_id.to_owned(),
            arena_bytes,
            n_ctx,
            max_blocks,
            num_layers,
            head_count_kv,
            head_dim,
            vocab_size,
            kv_dtype,
        };
        self.allocate_from_request(&req)
    }

    /// Deallocates an arena by ID, reclaiming its allocated memory budget.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if the arena ID is not registered.
    pub fn deallocate(&self, arena_id: u64) -> Result<u64, KvError> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| KvError::LockError(e.to_string()))?;
        let arena = guard
            .arenas
            .remove(&arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let freed_bytes = arena.len() as u64;
        guard.allocated_bytes = guard.allocated_bytes.saturating_sub(freed_bytes);
        Ok(freed_bytes)
    }

    /// Looks up an active arena by ID, returning a shared reference `Arc<SharedMemoryArena>`.
    #[must_use]
    pub fn get_arena(&self, arena_id: u64) -> Option<Arc<SharedMemoryArena>> {
        let guard = self.inner.read().ok()?;
        guard.arenas.get(&arena_id).cloned()
    }

    /// Obtains a duplicated file descriptor (`OwnedFd`) for the specified arena.
    ///
    /// Suitable for transfer via `SCM_RIGHTS` over Unix Domain Sockets.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if the arena is not found,
    /// or [`KvError::IpcError`] if `libc::dup` fails.
    pub fn get_arena_fd(&self, arena_id: u64) -> Result<OwnedFd, KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        arena
            .duplicate_fd()
            .map_err(|e| KvError::IpcError(e.to_string()))
    }

    /// Deallocates all registered arenas and resets allocated byte accounting to 0.
    pub fn clear(&self) -> usize {
        if let Ok(mut guard) = self.inner.write() {
            let count = guard.arenas.len();
            guard.arenas.clear();
            guard.allocated_bytes = 0;
            count
        } else {
            0
        }
    }

    /// Reads and validates the [`RecurrentStateSlotHeader`] for an active arena.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if `arena_id` is not registered, or
    /// [`KvError::InvalidDimensions`] / [`KvError::IpcError`] if the arena has no valid
    /// recurrent state region.
    pub fn read_recurrent_slot_header(
        &self,
        arena_id: u64,
    ) -> Result<RecurrentStateSlotHeader, KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let arena_bytes = arena.as_slice();
        let hdr_bytes = arena_bytes
            .get(..size_of::<ArenaHeader>())
            .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
        let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
            .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
        let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let region = arena_bytes.get(rec_off..).ok_or_else(|| {
            KvError::IpcError("arena truncated before recurrent region".to_owned())
        })?;
        read_recurrent_slot_header_from_region(region)
    }

    /// Reads the durably committed `active_bank` payload snapshot and header for crash recovery.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if `arena_id` is not registered, or
    /// [`KvError::InvalidDimensions`] / [`KvError::IpcError`] if the recurrent state region is invalid.
    pub fn read_active_recurrent_bank(
        &self,
        arena_id: u64,
    ) -> Result<(RecurrentStateSlotHeader, Vec<u8>), KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let arena_bytes = arena.as_slice();
        let hdr_bytes = arena_bytes
            .get(..size_of::<ArenaHeader>())
            .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
        let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
            .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
        let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let region = arena_bytes.get(rec_off..).ok_or_else(|| {
            KvError::IpcError("arena truncated before recurrent region".to_owned())
        })?;
        let rec_hdr = read_recurrent_slot_header_from_region(region)?;
        let total = usize::try_from(rec_hdr.total_region_bytes()).map_err(|_| {
            KvError::InvalidDimensions("recurrent region size exceeds address space".to_owned())
        })?;
        if region.len() < total {
            return Err(KvError::IpcError(
                "arena truncated inside recurrent ping-pong banks".to_owned(),
            ));
        }
        let bank_off =
            usize::try_from(rec_hdr.bank_offset_bytes(rec_hdr.active_bank)).map_err(|_| {
                KvError::InvalidDimensions("active bank offset exceeds address space".to_owned())
            })?;
        let bank_len = usize::try_from(rec_hdr.bank_stride_bytes).map_err(|_| {
            KvError::InvalidDimensions("bank stride exceeds address space".to_owned())
        })?;
        let bank_slice = &region[bank_off..bank_off + bank_len];
        Ok((rec_hdr, bank_slice.to_vec()))
    }
}

/// Computes the cacheline-aligned start offset of the ping-pong recurrent state region
/// deterministically immediately after the output logit slabs, without modifying [`ArenaHeader`].
#[must_use]
pub fn recurrent_region_offset(header: &ArenaHeader) -> u64 {
    let vocab_size = u64::from(header.vocab_size.max(1));
    let active_slots = u64::from(header.active_slots.max(1));
    let per_slab_bytes = vocab_size
        .saturating_mul(size_of::<f32>() as u64)
        .saturating_add(size_of::<OutputSlabHeader>() as u64);
    let aligned_slab_bytes = align_up_u64(per_slab_bytes, CACHELINE_ALIGNMENT as u64);
    let total_slabs_bytes = aligned_slab_bytes.saturating_mul(active_slots);
    let output_slabs_end = header.output_slabs_offset.saturating_add(total_slabs_bytes);
    align_up_u64(output_slabs_end, CACHELINE_ALIGNMENT as u64)
}

/// Reads and validates a [`RecurrentStateSlotHeader`] from the beginning of a recurrent region slice.
///
/// # Errors
/// Returns [`KvError::IpcError`] if `region` is shorter than 64 bytes or malformed, or
/// [`KvError::InvalidDimensions`] if header validation fails or `num_recurrent_layers == 0`.
pub fn read_recurrent_slot_header_from_region(
    region: &[u8],
) -> Result<RecurrentStateSlotHeader, KvError> {
    let hdr_bytes = region
        .get(..size_of::<RecurrentStateSlotHeader>())
        .ok_or_else(|| {
            KvError::IpcError("recurrent region shorter than 64-byte header".to_owned())
        })?;
    fence(Ordering::Acquire);
    let hdr = RecurrentStateSlotHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid RecurrentStateSlotHeader: {e}")))?;
    hdr.validate()
        .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
    if hdr.num_recurrent_layers == 0 {
        return Err(KvError::InvalidDimensions(
            "recurrent state header reports zero recurrent layers".to_owned(),
        ));
    }
    Ok(hdr)
}

/// Returns `(header, active_bank_readonly, inactive_bank_mutable)` for zero-tearing ping-pong execution.
///
/// Workers read checkpointed previous state from the active bank slice and write next-step state
/// exclusively into the inactive staging bank slice prior to calling [`commit_recurrent_step_in_region`].
///
/// # Errors
/// Returns [`KvError::IpcError`] or [`KvError::InvalidDimensions`] if the region is truncated or invalid.
pub fn recurrent_banks_from_region_mut(
    region: &mut [u8],
) -> Result<(RecurrentStateSlotHeader, &[u8], &mut [u8]), KvError> {
    let hdr = read_recurrent_slot_header_from_region(region)?;
    let hdr_len = size_of::<RecurrentStateSlotHeader>();
    let total = usize::try_from(hdr.total_region_bytes()).map_err(|_| {
        KvError::InvalidDimensions("recurrent region size exceeds address space".to_owned())
    })?;
    let stride = usize::try_from(hdr.bank_stride_bytes).map_err(|_| {
        KvError::InvalidDimensions("recurrent bank_stride exceeds address space".to_owned())
    })?;
    if region.len() < total {
        return Err(KvError::IpcError(format!(
            "recurrent region len ({}) < required ({total})",
            region.len()
        )));
    }
    let dual_payload = &mut region[hdr_len..total];
    let (first_bank, second_bank) = dual_payload.split_at_mut(stride);
    if hdr.active_bank == 0 {
        Ok((hdr, &*first_bank, second_bank))
    } else {
        Ok((hdr, &*second_bank, first_bank))
    }
}

/// Atomically flips `active_bank` (`0 <-> 1`) and updates `committed_step` in a recurrent region header.
///
/// Emits a `Release` barrier after staging-bank writes and stores `(new_step, inactive_bank)` in a
/// single aligned 64-bit atomic store (`AtomicU64`) at byte offset `0` of `region`, guaranteeing that
/// either both fields advance together or neither changes on crash.
///
/// # Errors
/// Returns [`KvError::IpcError`] if `region` is unaligned (`% 8 != 0`) or truncated, or
/// [`KvError::InvalidDimensions`] if header invariants fail.
pub fn commit_recurrent_step_in_region(
    region: &mut [u8],
    new_step: u32,
) -> Result<RecurrentStateSlotHeader, KvError> {
    let mut hdr = read_recurrent_slot_header_from_region(region)?;
    if (region.as_ptr() as usize) % align_of::<AtomicU64>() != 0 {
        return Err(KvError::IpcError(
            "recurrent region base pointer is not 8-byte aligned for atomic commit".to_owned(),
        ));
    }
    let next_bank = hdr.inactive_bank();
    hdr.committed_step = new_step;
    hdr.active_bank = next_bank;

    let mut word = [0_u8; 8];
    word[0..4].copy_from_slice(&new_step.to_ne_bytes());
    word[4..8].copy_from_slice(&next_bank.to_ne_bytes());
    let packed = u64::from_ne_bytes(word);

    fence(Ordering::Release);
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "8-byte AtomicU64 alignment is runtime-checked directly above"
    )]
    // SAFETY: `region` has len >= 64 and its base pointer was verified 8-byte aligned (`AtomicU64`).
    unsafe {
        AtomicU64::from_ptr(region.as_mut_ptr().cast::<u64>()).store(packed, Ordering::Release);
    }

    Ok(hdr)
}

/// Locates the recurrent region inside `arena_bytes` via [`ArenaHeader`] and returns split ping-pong views.
///
/// # Errors
/// Returns [`KvError`] if [`ArenaHeader`] or the trailing recurrent region is invalid or truncated.
pub fn recurrent_banks_in_arena_mut(
    arena_bytes: &mut [u8],
) -> Result<(RecurrentStateSlotHeader, &[u8], &mut [u8]), KvError> {
    let hdr_bytes = arena_bytes
        .get(..size_of::<ArenaHeader>())
        .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
    let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
    let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
        KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
    })?;
    let region = arena_bytes
        .get_mut(rec_off..)
        .ok_or_else(|| KvError::IpcError("arena truncated before recurrent region".to_owned()))?;
    recurrent_banks_from_region_mut(region)
}

/// Locates the recurrent region inside `arena_bytes` via [`ArenaHeader`] and commits `new_step`.
///
/// # Errors
/// Returns [`KvError`] if [`ArenaHeader`] or the trailing recurrent region is invalid or truncated.
pub fn commit_recurrent_step_in_arena(
    arena_bytes: &mut [u8],
    new_step: u32,
) -> Result<RecurrentStateSlotHeader, KvError> {
    let hdr_bytes = arena_bytes
        .get(..size_of::<ArenaHeader>())
        .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
    let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
    let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
        KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
    })?;
    let region = arena_bytes
        .get_mut(rec_off..)
        .ok_or_else(|| KvError::IpcError("arena truncated before recurrent region".to_owned()))?;
    commit_recurrent_step_in_region(region, new_step)
}

fn populate_arena_header(
    hdr: &mut ArenaHeader,
    total_bytes: u64,
    config: &ArenaAllocationConfig,
    layout: &ComputedArenaLayout,
) -> Result<(), KvError> {
    let mut reserved = [0_u8; 24];
    reserved[..4].copy_from_slice(&layout.seq_capacity.to_le_bytes());
    reserved[4] = config.kv_dtype;
    reserved[5] = u8::from(config.kv_latent_dim.is_some());

    *hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes,
        block_count: layout.block_count,
        tokens_per_block: BLOCK_SIZE_TOKENS,
        n_ctx: config.n_ctx,
        head_dim: config.head_dim,
        num_layers: config.num_layers,
        head_count_kv: config.head_count_kv.max(1),
        vocab_size: config.vocab_size.max(1),
        active_slots: config.active_slots.max(1),
        meta_table_offset: layout.meta_table_offset,
        token_slots_offset: layout.token_slots_offset,
        kv_blocks_offset: layout.kv_blocks_offset,
        output_slabs_offset: layout.output_slabs_offset,
        generation_counter: 1,
        swa_window_size: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        derived_weights_offset: 0,
        _reserved: reserved,
    };
    hdr.validate()
        .map_err(|msg| KvError::InvalidDimensions(msg.to_string()))
}

/// Initializes the `ArenaHeader`, `KvBlockMeta` entries, and slot/slab headers
/// within a newly mapped shared memory arena.
#[expect(
    clippy::too_many_lines,
    reason = "initializes all arena structures in single sequential pass"
)]
fn initialize_arena_contents(
    arena: &mut SharedMemoryArena,
    config: &ArenaAllocationConfig,
    layout: &ComputedArenaLayout,
) -> Result<(), KvError> {
    // 1. Initialize ArenaHeader at offset 0
    let total_bytes = arena.len() as u64;
    let hdr = arena
        .header_mut()
        .map_err(|e| KvError::IpcError(format!("failed to get header_mut: {e}")))?;
    populate_arena_header(hdr, total_bytes, config, layout)?;

    // 2. Initialize KvBlockMeta table (0..block_count)
    for idx in 0..layout.block_count as usize {
        let block_id = u32::try_from(idx)
            .map_err(|_| KvError::InvalidDimensions("block index exceeds u32 range".to_owned()))?;
        let meta = arena.block_meta_mut(idx).map_err(|e| {
            KvError::IpcError(format!("failed to initialize block_meta[{idx}]: {e}"))
        })?;
        *meta = KvBlockMeta {
            block_id,
            tag: TAG_FREE,
            _pad0: 0,
            refcount: 0,
            committed_token_count: 0,
            _pad1: 0,
            sampler_rng_state: [0; 4],
            padding: [0; 16],
        };
    }

    // 3. Initialize TokenInputSlotHeader for active_slots
    let active_slots = config.active_slots.max(1) as usize;
    let raw_slot_bytes = effective_token_slot_capacity(config) as usize * size_of::<u32>()
        + size_of::<TokenInputSlotHeader>();
    let per_slot_bytes = align_up_usize(raw_slot_bytes, CACHELINE_ALIGNMENT);
    let token_slots_base = usize::try_from(layout.token_slots_offset).map_err(|_| {
        KvError::InvalidDimensions("token_slots_offset exceeds address space".to_owned())
    })?;
    for slot_idx in 0..active_slots {
        let slot_base = token_slots_base + slot_idx * per_slot_bytes;
        let slot_slice = arena
            .slice_mut(slot_base, size_of::<TokenInputSlotHeader>())
            .map_err(|e| {
                KvError::IpcError(format!(
                    "failed to slice token slot header[{slot_idx}]: {e}"
                ))
            })?;
        let token_hdr = TokenInputSlotHeader::mut_from_bytes(slot_slice).map_err(|e| {
            KvError::IpcError(format!(
                "failed to decode TokenInputSlotHeader[{slot_idx}]: {e}"
            ))
        })?;
        *token_hdr = TokenInputSlotHeader {
            slot_seq_id: 0,
            token_len: 0,
            start_position: 0,
        };
    }

    // 4. Initialize OutputSlabHeader for active_slots
    let vocab_size = config.vocab_size.max(1);
    let raw_slab_bytes = vocab_size as usize * size_of::<f32>() + size_of::<OutputSlabHeader>();
    let per_slab_bytes = align_up_usize(raw_slab_bytes, CACHELINE_ALIGNMENT);
    let output_slabs_base = usize::try_from(layout.output_slabs_offset).map_err(|_| {
        KvError::InvalidDimensions("output_slabs_offset exceeds address space".to_owned())
    })?;
    for slot_idx in 0..active_slots {
        let slab_base = output_slabs_base + slot_idx * per_slab_bytes;
        let slab_slice = arena
            .slice_mut(slab_base, size_of::<OutputSlabHeader>())
            .map_err(|e| {
                KvError::IpcError(format!(
                    "failed to slice output slab header[{slot_idx}]: {e}"
                ))
            })?;
        let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).map_err(|e| {
            KvError::IpcError(format!(
                "failed to decode OutputSlabHeader[{slot_idx}]: {e}"
            ))
        })?;
        *slab_hdr = OutputSlabHeader {
            request_id: 0,
            vocab_size,
            position: 0,
            status: SLAB_STATUS_IDLE,
            _reserved: [0; 44],
        };
    }

    // 5. Initialize RecurrentStateSlotHeader and zero dual ping-pong banks when configured
    if config.num_recurrent_layers > 0 {
        let align = CACHELINE_ALIGNMENT as u64;
        let conv_aligned = u32::try_from(align_up_u64(
            u64::from(config.conv_state_bytes_per_layer),
            align,
        ))
        .map_err(|_| {
            KvError::InvalidDimensions("conv_state_bytes_per_layer exceeds u32".to_owned())
        })?;
        let ssm_aligned = u32::try_from(align_up_u64(
            u64::from(config.ssm_state_bytes_per_layer),
            align,
        ))
        .map_err(|_| {
            KvError::InvalidDimensions("ssm_state_bytes_per_layer exceeds u32".to_owned())
        })?;
        let rec_off = usize::try_from(layout.recurrent_state_offset).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let rec_total = usize::try_from(layout.recurrent_total_bytes).map_err(|_| {
            KvError::InvalidDimensions("recurrent_total_bytes exceeds address space".to_owned())
        })?;
        let rec_slice = arena.slice_mut(rec_off, rec_total).map_err(|e| {
            KvError::IpcError(format!("failed to slice recurrent state region: {e}"))
        })?;
        rec_slice.fill(0);
        let rec_hdr = RecurrentStateSlotHeader::new(
            config.num_recurrent_layers,
            conv_aligned,
            ssm_aligned,
            layout.recurrent_bank_stride_bytes,
        );
        rec_hdr
            .validate()
            .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
        let hdr_size = size_of::<RecurrentStateSlotHeader>();
        rec_slice[..hdr_size].copy_from_slice(rec_hdr.as_bytes());
    }

    Ok(())
}

/// Returns the exact per-slot sequence capacity (`blocks_per_slot * 16`) persisted in
/// [`ArenaHeader`], avoiding slab-padding drift.
#[must_use]
pub fn arena_seq_capacity(header: &ArenaHeader) -> u32 {
    let stored = u32::from_le_bytes([
        header._reserved[0],
        header._reserved[1],
        header._reserved[2],
        header._reserved[3],
    ]);
    if stored > 0 {
        return stored;
    }
    let slots = header.active_slots.max(1);
    (header.block_count / slots).saturating_mul(header.tokens_per_block)
}

/// Returns the KV cache element encoding (`KV_DTYPE_*`) recorded in [`ArenaHeader`].
#[must_use]
pub const fn arena_kv_dtype(header: &ArenaHeader) -> u8 {
    header._reserved[4]
}

/// Typed shared-memory handle exposing validated, bounds-checked views into an initialized arena.
#[derive(Debug)]
pub struct ArenaHandle<'a> {
    arena: &'a mut SharedMemoryArena,
}

impl<'a> ArenaHandle<'a> {
    /// Wraps a mapped [`SharedMemoryArena`] after validating its [`ArenaHeader`].
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] when the mapped buffer is smaller than an
    /// [`ArenaHeader`] or fails [`ArenaHeader::validate`].
    pub fn new(arena: &'a mut SharedMemoryArena) -> Result<Self, KvError> {
        {
            let (header, _) = ArenaHeader::ref_from_prefix(arena.as_slice()).map_err(|_| {
                KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
            })?;
            header
                .validate()
                .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;
        }
        Ok(Self { arena })
    }

    /// Returns the validated [`ArenaHeader`] at the base of the mapped arena.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if the header fails validation.
    pub fn header(&self) -> Result<&ArenaHeader, KvError> {
        let (header, _) = ArenaHeader::ref_from_prefix(self.arena.as_slice()).map_err(|_| {
            KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
        })?;
        header
            .validate()
            .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;
        Ok(header)
    }

    /// Returns a contiguous layer-major `[layer][K|V][pos][kv_dim]` mutable [`KvCacheView`] over
    /// the arena's initialized KV slab, bounded strictly to `seq_capacity` without trailing
    /// cacheline-padding drift.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if header validation fails, `kv_dtype != F32`,
    /// `active_slots != 1`, MLA latent compression is active, dimensions overflow `usize`, or
    /// the KV slab slice is unaligned or out of bounds.
    pub fn kv_cache_view_mut(&mut self) -> Result<KvCacheView<'_>, KvError> {
        kv_cache_view_from_arena_slice_mut(self.arena.as_mut_slice())
    }

    /// Consumes the handle and returns the underlying [`SharedMemoryArena`].
    #[must_use]
    pub fn into_inner(self) -> &'a mut SharedMemoryArena {
        self.arena
    }
}

/// Constructs a bounds-checked [`KvCacheView`] over an initialized arena byte buffer.
///
/// Exposed alongside [`ArenaHandle`] so layer-major `[layer][K|V][pos][kv_dim]` invariants can
/// also be verified directly under Miri without OS `memfd_create` / `mmap` syscalls.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if header validation fails, geometry or dtype checks
/// fail, or the exact KV slab range violates header region bounds or `f32` alignment.
pub fn kv_cache_view_from_arena_slice_mut(
    arena_bytes: &mut [u8],
) -> Result<KvCacheView<'_>, KvError> {
    let (header, _) = ArenaHeader::ref_from_prefix(arena_bytes).map_err(|_| {
        KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
    })?;
    header
        .validate()
        .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;

    let kv_dtype = arena_kv_dtype(header);
    if kv_dtype != KV_DTYPE_F32 {
        return Err(KvError::InvalidDimensions(format!(
            "KvCacheView requires F32 KV cache (kv_dtype=0), got {kv_dtype}"
        )));
    }
    if header._reserved[5] != 0 {
        return Err(KvError::InvalidDimensions(
            "KvCacheView does not support MLA latent-compressed KV slabs".to_owned(),
        ));
    }
    if header.num_layers == 0 || header.head_count_kv == 0 || header.head_dim == 0 {
        return Err(KvError::InvalidDimensions(
            "ArenaHeader KV geometry dimensions must be greater than zero".to_owned(),
        ));
    }
    let active_slots = header.active_slots.max(1);
    if active_slots != 1 {
        return Err(KvError::InvalidDimensions(format!(
            "KvCacheView expects single active slot, arena has active_slots={active_slots}"
        )));
    }
    let seq_cap_u32 = arena_seq_capacity(header);
    if seq_cap_u32 == 0 {
        return Err(KvError::InvalidDimensions(
            "ArenaHeader seq_capacity must be greater than zero".to_owned(),
        ));
    }

    let num_layers = usize::try_from(header.num_layers)
        .map_err(|_| KvError::InvalidDimensions("num_layers overflowed usize".to_owned()))?;
    let n_heads_kv = usize::try_from(header.head_count_kv)
        .map_err(|_| KvError::InvalidDimensions("head_count_kv overflowed usize".to_owned()))?;
    let head_dim = usize::try_from(header.head_dim)
        .map_err(|_| KvError::InvalidDimensions("head_dim overflowed usize".to_owned()))?;
    let seq_capacity = usize::try_from(seq_cap_u32)
        .map_err(|_| KvError::InvalidDimensions("seq_capacity overflowed usize".to_owned()))?;
    let sliding_window = if header.swa_window_size > 0 {
        Some(usize::try_from(header.swa_window_size).map_err(|_| {
            KvError::InvalidDimensions("swa_window_size overflowed usize".to_owned())
        })?)
    } else {
        None
    };

    let kv_offset = usize::try_from(header.kv_blocks_offset)
        .map_err(|_| KvError::InvalidDimensions("kv_blocks_offset overflowed usize".to_owned()))?;
    let output_offset = usize::try_from(header.output_slabs_offset).map_err(|_| {
        KvError::InvalidDimensions("output_slabs_offset overflowed usize".to_owned())
    })?;
    let total_bytes = usize::try_from(header.total_bytes)
        .map_err(|_| KvError::InvalidDimensions("total_bytes overflowed usize".to_owned()))?;

    let geometry = KvGeometry::new(
        num_layers,
        n_heads_kv,
        head_dim,
        seq_capacity,
        sliding_window,
    )
    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let required_f32s = geometry
        .required_f32_len()
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let required_bytes = required_f32s.checked_mul(size_of::<f32>()).ok_or_else(|| {
        KvError::InvalidDimensions("KV slab byte length overflowed usize".to_owned())
    })?;
    let kv_end = kv_offset.checked_add(required_bytes).ok_or_else(|| {
        KvError::InvalidDimensions("KV slab end offset overflowed usize".to_owned())
    })?;

    if kv_offset < size_of::<ArenaHeader>()
        || output_offset < kv_end
        || total_bytes > arena_bytes.len()
        || kv_end > arena_bytes.len()
    {
        return Err(KvError::InvalidDimensions(format!(
            "KV slab bounds [{kv_offset}..{kv_end}) invalid for output_slabs_offset={output_offset}, arena_len={}",
            arena_bytes.len()
        )));
    }

    let kv_bytes = &mut arena_bytes[kv_offset..kv_end];
    let kv_f32s = <[f32]>::mut_from_bytes(kv_bytes).map_err(|err| {
        KvError::InvalidDimensions(format!("KV slab byte slice conversion failed: {err:?}"))
    })?;
    KvCacheView::new(kv_f32s, geometry).map_err(|err| KvError::InvalidDimensions(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_allocation_and_header_initialization() {
        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("test_model", 64, 64, 2)
            .with_active_slots(1)
            .with_head_count_kv(2)
            .with_vocab_size(32000);

        let res = holder.allocate(&config).expect("allocation should succeed");
        assert!(res.arena_id > 0);
        assert!(res.total_bytes >= 256 * 1024);
        assert_eq!(res.block_count, 4); // 64 tokens / 16 = 4 blocks
        assert_eq!(res.model_id, "test_model");

        // Accounting verification
        assert_eq!(holder.total_allocated_bytes(), res.total_bytes);
        assert_eq!(holder.arena_count(), 1);
        assert!(holder.contains_arena(res.arena_id));

        // Arena inspection
        let arena = holder.get_arena(res.arena_id).expect("arena lookup");
        let hdr = arena.header().expect("header access");
        assert_eq!(hdr.magic, ARENA_MAGIC);
        assert_eq!(hdr.version, ARENA_VERSION);
        assert_eq!(hdr.block_count, 4);
        assert_eq!(hdr.tokens_per_block, 16);
        assert_eq!(hdr.n_ctx, 64);
        assert_eq!(hdr.head_dim, 64);
        assert_eq!(hdr.num_layers, 2);
        assert_eq!(hdr.head_count_kv, 2);
        assert_eq!(hdr.vocab_size, 32000);
        assert_eq!(hdr.active_slots, 1);
        assert!(hdr.validate().is_ok());

        // Verify KvBlockMeta entries initialized to free
        for i in 0..4 {
            let meta = arena.block_meta(i).expect("block meta");
            assert_eq!(meta.block_id, u32::try_from(i).unwrap());
            assert_eq!(meta.tag, TAG_FREE);
            assert_eq!(meta.refcount, 0);
            assert_eq!(meta.committed_tokens(), 0);
        }
    }

    #[test]
    fn test_memlock_ceiling_rejection_single_arena() {
        // Configure small memlock ceiling (64 KiB)
        let small_ceiling = 64 * 1024;
        let holder = KvHolder::with_memlock_ceiling(small_ceiling);

        // Arena with n_ctx=64 requires ~256 KiB, exceeding 64 KiB ceiling
        let config = ArenaAllocationConfig::new("oversized_model", 64, 64, 2);
        let res = holder.allocate(&config);

        match res {
            Err(
                err @ KvError::InsufficientLockableMemory {
                    required_bytes,
                    allocated_bytes,
                    memlock_ceiling_bytes,
                },
            ) => {
                assert!(required_bytes > small_ceiling);
                assert_eq!(allocated_bytes, 0);
                assert_eq!(memlock_ceiling_bytes, small_ceiling);
                assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
                assert_eq!(err.exit_code(), 11);
            }
            other => panic!("Expected InsufficientLockableMemory, got {other:?}"),
        }

        // Verify no budget leaked
        assert_eq!(holder.total_allocated_bytes(), 0);
        assert_eq!(holder.arena_count(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_memlock_ceiling_rejection_multi_arena_cumulative() {
        let config = ArenaAllocationConfig::new("model_a", 64, 64, 2);
        let layout = compute_arena_layout(&config).expect("layout");
        let single_arena_bytes = layout.total_bytes;

        // Ceiling sized to hold exactly 1 arena, but not 2
        let tight_ceiling = single_arena_bytes + (single_arena_bytes / 2);
        let holder = KvHolder::with_memlock_ceiling(tight_ceiling);

        // First allocation succeeds
        let res1 = holder.allocate(&config).expect("first allocation");
        assert_eq!(holder.total_allocated_bytes(), single_arena_bytes);

        // Second allocation must fail with exit code 11
        let config2 = ArenaAllocationConfig::new("model_b", 64, 64, 2);
        let res2 = holder.allocate(&config2);

        match res2 {
            Err(
                err @ KvError::InsufficientLockableMemory {
                    allocated_bytes,
                    memlock_ceiling_bytes,
                    ..
                },
            ) => {
                assert_eq!(allocated_bytes, single_arena_bytes);
                assert_eq!(memlock_ceiling_bytes, tight_ceiling);
                assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
                assert_eq!(err.exit_code(), 11);
            }
            other => panic!("Expected InsufficientLockableMemory, got {other:?}"),
        }

        // Verify allocated bytes remains strictly single_arena_bytes
        assert_eq!(holder.total_allocated_bytes(), single_arena_bytes);

        // Deallocate first arena
        let freed = holder.deallocate(res1.arena_id).expect("deallocate");
        assert_eq!(freed, single_arena_bytes);
        assert_eq!(holder.total_allocated_bytes(), 0);

        // Now second allocation succeeds cleanly
        let res3 = holder
            .allocate(&config2)
            .expect("second allocation succeeds after freed");
        assert_eq!(holder.total_allocated_bytes(), res3.total_bytes);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_lookup_and_fd_duplication() {
        use std::os::fd::AsRawFd;

        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("model_fd", 64, 64, 2);
        let res = holder.allocate(&config).expect("allocation");

        // Lookup existing
        assert!(holder.get_arena(res.arena_id).is_some());
        let fd = holder.get_arena_fd(res.arena_id).expect("fd dup");
        assert!(fd.as_raw_fd() >= 0);

        // Lookup non-existent
        assert!(holder.get_arena(9999).is_none());
        assert_eq!(
            holder.get_arena_fd(9999).unwrap_err(),
            KvError::ArenaNotFound(9999)
        );
        assert_eq!(
            holder.get_arena_fd(9999).unwrap_err().status_code(),
            StatusCode::ArenaNotFound
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_deallocation_and_clear() {
        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("model_clear", 64, 64, 2);
        let res1 = holder.allocate(&config).expect("alloc 1");
        let res2 = holder.allocate(&config).expect("alloc 2");
        assert_eq!(holder.arena_count(), 2);

        // Single deallocation
        let freed = holder.deallocate(res1.arena_id).expect("dealloc 1");
        assert_eq!(freed, res1.total_bytes);
        assert_eq!(holder.arena_count(), 1);
        assert!(!holder.contains_arena(res1.arena_id));
        assert!(holder.contains_arena(res2.arena_id));

        // Clear all
        let cleared = holder.clear();
        assert_eq!(cleared, 1);
        assert_eq!(holder.arena_count(), 0);
        assert_eq!(holder.total_allocated_bytes(), 0);
    }

    #[test]
    fn test_invalid_dimensions_rejected() {
        let holder = KvHolder::new();

        // n_ctx = 0
        let mut cfg = ArenaAllocationConfig::new("m", 0, 64, 2);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );

        // head_dim = 0
        cfg = ArenaAllocationConfig::new("m", 64, 0, 2);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );

        // num_layers = 0
        cfg = ArenaAllocationConfig::new("m", 64, 64, 0);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );
    }

    #[test]
    fn test_compute_hybrid_kv_block_count() {
        assert_eq!(compute_hybrid_kv_block_count(0, &[]), 0);

        // Gemma 3 / 4: 36 layers in 5:1 local(1024) : global(n_ctx) pattern -> 30 local + 6 global.
        let gemma3_policies: Vec<LayerKvPolicy> = (0..36)
            .map(|l| {
                if (l + 1) % 6 == 0 {
                    LayerKvPolicy::GlobalFullContext
                } else {
                    LayerKvPolicy::SlidingWindow {
                        window_tokens: 1024,
                    }
                }
            })
            .collect();
        // 30 * (1024 / 16) + 6 * (32768 / 16) = 30 * 64 + 6 * 2048 = 1920 + 12288 = 14208.
        assert_eq!(
            compute_hybrid_kv_block_count(32_768, &gemma3_policies),
            14_208
        );

        // Gemma 2: 26 layers in 1:1 alternating local(4096) : global(8192).
        let gemma2_policies: Vec<LayerKvPolicy> = (0..26)
            .map(|l| {
                if l % 2 == 0 {
                    LayerKvPolicy::SlidingWindow {
                        window_tokens: 4096,
                    }
                } else {
                    LayerKvPolicy::GlobalFullContext
                }
            })
            .collect();
        // 13 * 256 + 13 * 512 = 9984.
        assert_eq!(
            compute_hybrid_kv_block_count(8_192, &gemma2_policies),
            9_984
        );

        // Arena layout with hybrid policies allocates strictly fewer bytes than homogeneous layout.
        let full_cfg = ArenaAllocationConfig::new("gemma3", 32_768, 256, 36).with_head_count_kv(4);
        let hybrid_cfg = ArenaAllocationConfig::new("gemma3", 32_768, 256, 36)
            .with_head_count_kv(4)
            .with_layer_policies(gemma3_policies);
        let full_layout = compute_arena_layout(&full_cfg).expect("full layout");
        let hybrid_layout = compute_arena_layout(&hybrid_cfg).expect("hybrid layout");
        assert!(
            hybrid_layout.total_bytes * 4 < full_layout.total_bytes,
            "hybrid 5:1 arena ({}) should be >4x smaller than homogeneous full-context ({})",
            hybrid_layout.total_bytes,
            full_layout.total_bytes
        );
    }

    #[test]
    fn test_holder_mla_latent_stride() {
        let cfg = ArenaAllocationConfig::new("deepseek2", 64, 192, 27)
            .with_head_count_kv(16)
            .with_kv_latent_dim(512, 64);
        assert_eq!(cfg.kv_latent_dim, Some(576));
        let layout = compute_arena_layout(&cfg).expect("mla layout");
        let zero_cfg = ArenaAllocationConfig::new("deepseek2", 64, 0, 1).with_kv_latent_dim(0, 0);
        assert!(matches!(
            compute_arena_layout(&zero_cfg),
            Err(KvError::InvalidDimensions(_))
        ));
        assert!(layout.total_bytes > 0);
    }

    #[test]
    fn test_mla_compressed_block_bytes() {
        // Single layer, single 16-token block: 16 * (512 + 64) * 4 = 36_864 bytes of KV block payload
        let cfg = ArenaAllocationConfig::new("deepseek2", 16, 0, 1)
            .with_vocab_size(0)
            .with_kv_latent_dim(512, 64);
        let layout = compute_arena_layout(&cfg).expect("single-block mla layout");
        let kv_span = layout.output_slabs_offset - layout.kv_blocks_offset;
        assert_eq!(kv_span, 36_864);
    }

    #[test]
    fn test_deepseek2_mla_latent_kv_layout() {
        // DeepSeek-V2-Lite: 27 layers, n_ctx = 4096, uncompressed 16*192 vs latent 512+64=576
        let uncompressed_cfg =
            ArenaAllocationConfig::new("deepseek2", 4096, 192, 27).with_head_count_kv(16);
        let mla_cfg = ArenaAllocationConfig::new("deepseek2", 4096, 192, 27)
            .with_head_count_kv(16)
            .with_kv_latent_dim(512, 64);
        let full_layout = compute_arena_layout(&uncompressed_cfg).expect("full layout");
        let mla_layout = compute_arena_layout(&mla_cfg).expect("mla layout");
        assert!(
            mla_layout.total_bytes * 5 < full_layout.total_bytes,
            "MLA latent arena ({}) should be >80% smaller than uncompressed GQA ({})",
            mla_layout.total_bytes,
            full_layout.total_bytes
        );
    }

    #[test]
    fn test_o1_recurrent_state_arena_sizing() {
        let small_ctx_cfg =
            ArenaAllocationConfig::new_recurrent("mamba2", 512, 32_000, 64, 1024, 4096);
        let large_ctx_cfg =
            ArenaAllocationConfig::new_recurrent("mamba2", 32_768, 32_000, 64, 1024, 4096);

        let small_layout = compute_arena_layout(&small_ctx_cfg).expect("512-ctx recurrent layout");
        let large_layout = compute_arena_layout(&large_ctx_cfg).expect("32k-ctx recurrent layout");

        assert_eq!(small_layout.block_count, 0);
        assert_eq!(large_layout.block_count, 0);
        assert_eq!(
            small_layout.total_bytes, large_layout.total_bytes,
            "pure recurrent arena sizing must remain O(1) as n_ctx grows from 512 to 32768"
        );
        assert_eq!(small_layout.recurrent_bank_stride_bytes, 64 * (1024 + 4096));
        assert_eq!(small_layout.recurrent_total_bytes, 64 + 2 * 64 * 5120);

        let hybrid_cfg = ArenaAllocationConfig::new("jamba", 1024, 128, 4)
            .with_head_count_kv(4)
            .with_vocab_size(32_000)
            .with_recurrent_state(28, 512, 2048);
        let hybrid_layout = compute_arena_layout(&hybrid_cfg).expect("hybrid jamba layout");
        assert_eq!(hybrid_layout.block_count, 64);
        assert!(hybrid_layout.recurrent_state_offset >= hybrid_layout.output_slabs_offset);
        assert_eq!(hybrid_layout.recurrent_bank_stride_bytes, 28 * (512 + 2048));
        assert_eq!(hybrid_layout.recurrent_total_bytes, 64 + 2 * 28 * 2560);
    }

    #[test]
    fn test_ping_pong_bank_flip_and_rollback() {
        #[repr(C, align(64))]
        struct AlignedRegion([u8; 4096]);

        let mut region = AlignedRegion([0_u8; 4096]);
        let init = RecurrentStateSlotHeader::new(2, 64, 128, 384);
        assert_eq!(init.bank_stride_bytes, 384);
        assert_eq!(init.total_region_bytes(), 832);
        region.0[..size_of::<RecurrentStateSlotHeader>()].copy_from_slice(init.as_bytes());

        {
            let (hdr, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("initial bank split");
            assert_eq!(hdr.committed_step, 0);
            assert_eq!(hdr.active_bank, 0);
            assert!(active.iter().all(|&b| b == 0));
            inactive.fill(0xA1);
        }

        let step_one_hdr =
            commit_recurrent_step_in_region(&mut region.0, 1).expect("commit step 1");
        assert_eq!(step_one_hdr.committed_step, 1);
        assert_eq!(step_one_hdr.active_bank, 1);

        // Simulate a worker crash partway through writing the staging bank before atomic commit.
        {
            let (hdr, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("step 1 bank split");
            assert_eq!(hdr.committed_step, 1);
            assert_eq!(hdr.active_bank, 1);
            assert!(active.iter().all(|&b| b == 0xA1));
            inactive[..128].fill(0xEE);
        }

        // Rollback verification: committed_step, active_bank, and bank 1 contents remain intact.
        let reread = read_recurrent_slot_header_from_region(&region.0).expect("reread header");
        assert_eq!(reread.committed_step, 1);
        assert_eq!(reread.active_bank, 1);
        {
            let (_, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("post-crash bank split");
            assert!(
                active.iter().all(|&b| b == 0xA1),
                "active committed bank must never tear on incomplete staging write"
            );
            inactive.fill(0xB2);
        }

        let step_two_hdr =
            commit_recurrent_step_in_region(&mut region.0, 2).expect("commit step 2");
        assert_eq!(step_two_hdr.committed_step, 2);
        assert_eq!(step_two_hdr.active_bank, 0);
        let (hdr_after, active_after, _) =
            recurrent_banks_from_region_mut(&mut region.0).expect("step 2 bank split");
        assert_eq!(hdr_after.committed_step, 2);
        assert_eq!(hdr_after.active_bank, 0);
        assert!(active_after.iter().all(|&b| b == 0xB2));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_ping_pong_recurrent_checkpoint() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new_recurrent("rwkv7", 256, 1024, 4, 128, 256);
        let res = holder
            .allocate(&cfg)
            .expect("pure recurrent allocation should succeed");
        assert_eq!(res.block_count, 0);

        let initial_hdr = holder
            .read_recurrent_slot_header(res.arena_id)
            .expect("recurrent header should be initialized");
        assert_eq!(initial_hdr.committed_step, 0);
        assert_eq!(initial_hdr.active_bank, 0);
        assert_eq!(initial_hdr.num_recurrent_layers, 4);
        assert_eq!(initial_hdr.conv_state_bytes_per_layer, 128);
        assert_eq!(initial_hdr.ssm_state_bytes_per_layer, 256);

        let fd = holder.get_arena_fd(res.arena_id).expect("arena fd");
        let len = usize::try_from(res.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_rec_rw", false)
            .expect("rw mapping from existing fd");

        {
            let (hdr, active, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("arena bank split");
            assert_eq!(hdr.active_bank, 0);
            assert!(active.iter().all(|&b| b == 0));
            inactive.fill(0x3C);
        }

        let committed =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 7).expect("commit step 7");
        assert_eq!(committed.committed_step, 7);
        assert_eq!(committed.active_bank, 1);

        let (snap_hdr, snap_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("read committed active bank");
        assert_eq!(snap_hdr.committed_step, 7);
        assert_eq!(snap_hdr.active_bank, 1);
        assert_eq!(
            snap_bank.len(),
            usize::try_from(snap_hdr.bank_stride_bytes).unwrap()
        );
        assert!(snap_bank.iter().all(|&b| b == 0x3C));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_atomic_ping_pong_recurrent_recovery() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new_recurrent("mamba2", 512, 2048, 2, 64, 192);
        let res = holder
            .allocate(&cfg)
            .expect("recurrent arena allocation should succeed");
        let fd = holder.get_arena_fd(res.arena_id).expect("arena fd");
        let len = usize::try_from(res.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_rec_recovery", false)
            .expect("rw mapping");

        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split step 1");
            inactive.fill(0x11);
        }
        let first_commit =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 1).expect("commit step 1");
        assert_eq!(first_commit.committed_step, 1);
        assert_eq!(first_commit.active_bank, 1);

        // Write a partial torn state to inactive bank 0 and simulate crash before commit.
        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split torn");
            inactive[..64].fill(0xFF);
        }

        let (recovered_hdr, recovered_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("recovery snapshot");
        assert_eq!(recovered_hdr.committed_step, 1);
        assert_eq!(recovered_hdr.active_bank, 1);
        assert!(recovered_bank.iter().all(|&b| b == 0x11));

        // Resume cleanly and commit step 2.
        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split step 2");
            inactive.fill(0x22);
        }
        let second_commit =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 2).expect("commit step 2");
        assert_eq!(second_commit.committed_step, 2);
        assert_eq!(second_commit.active_bank, 0);
        let (final_hdr, final_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("final snapshot");
        assert_eq!(final_hdr.committed_step, 2);
        assert_eq!(final_hdr.active_bank, 0);
        assert!(final_bank.iter().all(|&b| b == 0x22));
    }

    #[test]
    fn test_from_request_hydrates_real_geometry_and_fails_closed() {
        let req = AllocateArenaRequest {
            model_id: "qwen2.5-7b-instruct".to_owned(),
            arena_bytes: 16 * 1024 * 1024,
            n_ctx: 2048,
            max_blocks: 128,
            num_layers: 28,
            head_count_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            kv_dtype: KV_DTYPE_F16,
        };
        let cfg =
            ArenaAllocationConfig::from_request(&req).expect("valid request hydrates cleanly");
        assert_eq!(cfg.model_id, "qwen2.5-7b-instruct");
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.max_blocks, Some(128));
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.head_count_kv, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 128_256);
        assert_eq!(cfg.kv_dtype, KV_DTYPE_F16);
        assert_eq!(cfg.explicit_bytes, Some(16 * 1024 * 1024));

        // Fail closed on every zeroed geometry/capacity dimension and invalid kv_dtype.
        let invalid_cases = [
            AllocateArenaRequest {
                model_id: "   ".to_owned(),
                ..req.clone()
            },
            AllocateArenaRequest {
                arena_bytes: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                n_ctx: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                max_blocks: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                num_layers: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                head_count_kv: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                head_dim: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                vocab_size: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                kv_dtype: 99,
                ..req
            },
        ];

        for invalid in &invalid_cases {
            let err = ArenaAllocationConfig::from_request(invalid)
                .expect_err("zero/invalid dimension must fail closed");
            assert!(
                matches!(err, KvError::InvalidDimensions(_)),
                "expected InvalidDimensions, got {err:?}"
            );
        }
    }

    #[test]
    fn test_compute_arena_layout_rejects_zero_kv_dims_and_bad_dtype() {
        let zero_heads = ArenaAllocationConfig::new("m", 64, 64, 2).with_head_count_kv(0);
        assert!(matches!(
            compute_arena_layout(&zero_heads),
            Err(KvError::InvalidDimensions(_))
        ));

        let zero_vocab = ArenaAllocationConfig::new("m", 64, 64, 2).with_vocab_size(0);
        assert!(matches!(
            compute_arena_layout(&zero_vocab),
            Err(KvError::InvalidDimensions(_))
        ));

        let bad_dtype = ArenaAllocationConfig::new("m", 64, 64, 2).with_kv_dtype(3);
        assert!(matches!(
            compute_arena_layout(&bad_dtype),
            Err(KvError::InvalidDimensions(_))
        ));

        // Verify F16 layout allocates 2 bytes per KV scalar instead of 4.
        let f32_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_F32);
        let f16_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_F16);
        let f32_layout = compute_arena_layout(&f32_cfg).expect("f32 layout");
        let f16_layout = compute_arena_layout(&f16_cfg).expect("f16 layout");
        assert_eq!(f32_layout.seq_capacity, 32);
        assert_eq!(f16_layout.seq_capacity, 32);
        let f32_span = f32_layout.output_slabs_offset - f32_layout.kv_blocks_offset;
        let f16_span = f16_layout.output_slabs_offset - f16_layout.kv_blocks_offset;
        assert_eq!(f32_span, 4 * 2 * 32 * 4 * 64 * 4);
        assert_eq!(f16_span, 4 * 2 * 32 * 4 * 64 * 2);

        // Verify Q8_0 layout uses 34 bytes per 32-element group.
        let q8_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_Q8_0);
        let q8_layout = compute_arena_layout(&q8_cfg).expect("q8_0 layout");
        let kv_dim = 4_u64 * 64;
        let expected_q8_raw = 4 * 2 * 32 * (kv_dim / 32) * 34;
        assert_eq!(
            q8_layout.output_slabs_offset - q8_layout.kv_blocks_offset,
            align_up_u64(expected_q8_raw, CACHELINE_ALIGNMENT as u64)
        );
    }

    #[test]
    fn test_layer_major_kv_cache_view_writes_and_reads_without_seq_cap_drift() {
        #[repr(C, align(64))]
        #[derive(Clone, Copy, FromBytes, IntoBytes, zerocopy::Immutable)]
        struct AlignedLine([u8; 64]);

        // n_ctx = 20 -> 2 blocks * 16 = seq_capacity 32 (non-exact n_ctx vs block padding).
        // Add an extra 256-byte tail beyond total_bytes to verify view length has zero drift.
        let cfg = ArenaAllocationConfig::new("qwen-layer-major", 20, 4, 3)
            .with_head_count_kv(2)
            .with_vocab_size(64)
            .with_kv_dtype(KV_DTYPE_F32);
        let layout = compute_arena_layout(&cfg).expect("layer-major layout");
        assert_eq!(layout.block_count, 2);
        assert_eq!(layout.seq_capacity, 32);

        let total_usize = usize::try_from(layout.total_bytes).expect("total_bytes fits usize");
        let lines = (total_usize + 256).div_ceil(64);
        let mut backing = vec![AlignedLine([0_u8; 64]); lines];
        let arena_bytes = backing.as_mut_bytes();

        {
            let (hdr_mut, _) =
                ArenaHeader::mut_from_prefix(&mut arena_bytes[..total_usize]).expect("header");
            populate_arena_header(hdr_mut, layout.total_bytes, &cfg, &layout)
                .expect("populate ArenaHeader");
        }

        {
            let (hdr, _) = ArenaHeader::ref_from_prefix(arena_bytes).expect("header");
            assert_eq!(arena_seq_capacity(hdr), 32);
            assert_eq!(arena_kv_dtype(hdr), KV_DTYPE_F32);
            assert_eq!(hdr.n_ctx, 20);
            assert_eq!(hdr.num_layers, 3);
            assert_eq!(hdr.head_count_kv, 2);
            assert_eq!(hdr.head_dim, 4);
            assert_eq!(hdr.vocab_size, 64);
        }

        {
            let mut view = kv_cache_view_from_arena_slice_mut(arena_bytes)
                .expect("construct contiguous layer-major KvCacheView");
            let geom = view.geometry();
            assert_eq!(geom.num_layers, 3);
            assert_eq!(geom.n_heads_kv, 2);
            assert_eq!(geom.head_dim, 4);
            assert_eq!(geom.seq_capacity, 32);
            assert_eq!(view.kv_dim(), 8);

            // Populate positions 0..=20 in every layer with deterministic markers.
            for layer in 0..3 {
                for pos in 0..=20 {
                    let (k_slot, v_slot) = view
                        .write_pos_range(layer, pos)
                        .expect("write within seq_capacity");
                    for d in 0..8 {
                        let code = u16::try_from(layer * 1000 + pos * 10 + d).unwrap();
                        let marker = f32::from(code);
                        k_slot[d] = marker;
                        v_slot[d] = -marker - 0.5;
                    }
                }
            }

            // Read back contiguous 0..=20 range (`end_pos = 21`) across each layer.
            for layer in 0..3 {
                let (k_valid, v_valid) = view
                    .valid_kv_range(layer, 21)
                    .expect("valid range up to pos 20");
                assert_eq!(k_valid.len(), 21 * 8);
                assert_eq!(v_valid.len(), 21 * 8);
                for pos in 0..=20 {
                    let base = pos * 8;
                    for d in 0..8 {
                        let code = u16::try_from(layer * 1000 + pos * 10 + d).unwrap();
                        let expected_k = f32::from(code);
                        let expected_v = -expected_k - 0.5;
                        assert_eq!(k_valid[base + d], expected_k);
                        assert_eq!(v_valid[base + d], expected_v);
                    }
                }

                // Remaining positions 21..32 in plane stay zeroed without bleed.
                let (k_plane, v_plane) = view.layer_kv(layer).expect("layer_kv");
                assert_eq!(k_plane.len(), 32 * 8);
                assert!(k_plane[21 * 8..].iter().all(|&x| x == 0.0));
                assert!(v_plane[21 * 8..].iter().all(|&x| x == 0.0));
            }
        }

        // Verify fail-closed rejection on non-F32 KV dtype, multi-slot, and MLA flag.
        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut._reserved[4] = KV_DTYPE_F16;
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());

        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut._reserved[4] = KV_DTYPE_F32;
        hdr_mut.active_slots = 2;
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());

        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut.active_slots = 1;
        hdr_mut._reserved[5] = 1;
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_handle_real_model_geometry_fd_roundtrip() {
        let holder = KvHolder::new();
        let req = AllocateArenaRequest {
            model_id: "qwen2.5-7b-fd-roundtrip".to_owned(),
            arena_bytes: 16 * 1024 * 1024,
            n_ctx: 20,
            max_blocks: 2,
            num_layers: 28,
            head_count_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            kv_dtype: KV_DTYPE_F32,
        };
        let alloc = holder
            .allocate_from_request(&req)
            .expect("real geometry arena allocation");
        let fd = holder.get_arena_fd(alloc.arena_id).expect("arena fd");
        let len = usize::try_from(alloc.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_kv_view", false)
            .expect("rw arena mapping");

        let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle validates header");
        let hdr = handle.header().expect("header");
        assert_eq!(hdr.num_layers, 28);
        assert_eq!(hdr.head_count_kv, 8);
        assert_eq!(hdr.head_dim, 128);
        assert_eq!(hdr.vocab_size, 128_256);
        assert_eq!(hdr.n_ctx, 20);
        assert_eq!(arena_seq_capacity(hdr), 32);

        let mut view = handle
            .kv_cache_view_mut()
            .expect("kv_cache_view_mut returns contiguous view");
        assert_eq!(view.geometry().num_layers, 28);
        assert_eq!(view.geometry().seq_capacity, 32);
        assert_eq!(view.kv_dim(), 1024);

        for pos in 0..=20 {
            let val = f32::from(u16::try_from(pos).unwrap()) + 1.0;
            let (k, v) = view.write_pos_range(27, pos).expect("write layer 27");
            k.fill(val);
            v.fill(-val);
        }
        let (k_valid, v_valid) = view.valid_kv_range(27, 21).expect("read layer 27 0..=20");
        assert_eq!(k_valid.len(), 21 * 1024);
        assert_eq!(v_valid.len(), 21 * 1024);
        assert_eq!(k_valid[20 * 1024], 21.0);
        assert_eq!(v_valid[20 * 1024], -21.0);
    }
}
