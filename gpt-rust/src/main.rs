// ============================================================================
// GPT TRANSFORMER WITH GROUPED QUERY ATTENTION (GQA)
// Vectorized Tensor Autodiff Engine — Rust (2026)
// CPU-optimized: matrixmultiply (BLAS-level), Rayon (parallel heads), SIMD-friendly
//
// Mirrors @karpathy's pure-Python GPT but replaces the scalar autograd with a
// vectorized tensor-tape engine identical in design to the RNN-Rust project:
//   • Each graph node = dense row-major matrix, NOT a per-scalar node
//   • matrixmultiply::sgemm for all matmuls (forward + backward)
//   • Arena-style graph with freeze_params() / reset()
//   • Rayon par_iter for multi-head attention
//   • Tight iterator loops that auto-vectorize to SIMD under -C opt-level=3
//   • BPE (Byte Pair Encoding) tokenizer — trained separately, stored as JSON
//   • RoPE (Rotary Position Embedding) — no learned positional embeddings
//   • Grouped Query Attention: n_kv_head KV heads shared across n_head Q heads
//     — reduces KV params & compute while preserving quality
//   • Mixture of Experts (MoE): shared always-on expert + top-k sparse routed
//     experts per token — DeepSeek-style gating with SwiGLU experts
// ============================================================================

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Clone)]
struct Config {
    // Model
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    n_layer: usize,
    block_size: usize,
    n_experts: usize,  // total experts (expert 0 is always-on)
    top_k: usize,      // how many sparse experts to activate per token
    // Training
    num_steps: usize,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
    weight_decay: f32,
    // Inference
    temperature: f32,
    num_samples: usize,
    gen_length: usize,
    // Data
    training_file: String,
    // Tokenizer
    tokenizer_path: String,
    target_vocab_size: usize,  // target BPE vocab size for train-tokenizer
    // Checkpoint
    save_path: String,
    load_checkpoint: String,
    save_every: usize,
    save_on_complete: bool,
    log_every: usize,
    // Command
    command: String,
}

impl Config {
    fn default_config() -> Self {
        Config {
            n_embd: 16,
            n_head: 4,
            n_kv_head: 2,
            n_layer: 1,
            block_size: 16,
            n_experts: 4,
            top_k: 2,
            num_steps: 1000,
            learning_rate: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps_adam: 1e-8,
            weight_decay: 0.0,
            temperature: 0.5,
            num_samples: 20,
            gen_length: 0, // 0 = use block_size
            training_file: "input.txt".to_string(),
            tokenizer_path: "./tokenizer.json".to_string(),
            target_vocab_size: 512,
            save_path: "./gpt_checkpoint.bin".to_string(),
            load_checkpoint: String::new(),
            save_every: 0,
            save_on_complete: true,
            log_every: 1,
            command: String::new(),
        }
    }

    fn from_args() -> Self {
        let mut c = Self::default_config();
        for arg in env::args().skip(1) {
            if !arg.starts_with("--") {
                if c.command.is_empty() {
                    c.command = arg.to_lowercase();
                }
                continue;
            }
            let arg = arg.trim_start_matches("--");
            if let Some((key, val)) = arg.split_once('=') {
                match key.to_lowercase().as_str() {
                    "nembd" | "n_embd" => c.n_embd = val.parse().unwrap_or(c.n_embd),
                    "nhead" | "n_head" => c.n_head = val.parse().unwrap_or(c.n_head),
                    "nkvhead" | "n_kv_head" => c.n_kv_head = val.parse().unwrap_or(c.n_kv_head),
                    "nlayer" | "n_layer" => c.n_layer = val.parse().unwrap_or(c.n_layer),
                    "blocksize" | "block_size" => c.block_size = val.parse().unwrap_or(c.block_size),
                    "nexperts" | "n_experts" => c.n_experts = val.parse().unwrap_or(c.n_experts),
                    "topk" | "top_k" => c.top_k = val.parse().unwrap_or(c.top_k),
                    "numsteps" | "num_steps" => c.num_steps = val.parse().unwrap_or(c.num_steps),
                    "learningrate" | "lr" => c.learning_rate = val.parse().unwrap_or(c.learning_rate),
                    "beta1" => c.beta1 = val.parse().unwrap_or(c.beta1),
                    "beta2" => c.beta2 = val.parse().unwrap_or(c.beta2),
                    "weightdecay" | "wd" => c.weight_decay = val.parse().unwrap_or(c.weight_decay),
                    "temperature" | "temp" => c.temperature = val.parse().unwrap_or(c.temperature),
                    "numsamples" => c.num_samples = val.parse().unwrap_or(c.num_samples),
                    "genlength" => c.gen_length = val.parse().unwrap_or(c.gen_length),
                    "trainingfile" => c.training_file = val.to_string(),
                    "tokenizerpath" | "tokenizer_path" | "tokenizer" => c.tokenizer_path = val.to_string(),
                    "targetvocabsize" | "target_vocab_size" | "vocabsize" | "vocab_size" => c.target_vocab_size = val.parse().unwrap_or(c.target_vocab_size),
                    "savepath" => c.save_path = val.to_string(),
                    "loadcheckpoint" => c.load_checkpoint = val.to_string(),
                    "saveevery" => c.save_every = val.parse().unwrap_or(c.save_every),
                    "saveoncomplete" => c.save_on_complete = val == "true",
                    "logevery" => c.log_every = val.parse().unwrap_or(c.log_every),
                    _ => {}
                }
            }
        }
        c
    }
}

// ============================================================================
// SIMD-FRIENDLY VECTOR OPERATIONS
// Tight iterator loops auto-vectorize with rustc -C opt-level=3 + LTO
// ============================================================================

#[inline]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline]
fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; }
}

#[inline]
fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[inline]
fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

#[inline]
fn vec_scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

#[inline]
fn vec_scale_inplace(a: &mut [f32], s: f32) {
    for x in a.iter_mut() { *x *= s; }
}

#[inline]
fn vec_max(a: &[f32]) -> f32 {
    a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[inline]
fn vec_exp(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| x.exp()).collect()
}

// ============================================================================
// VECTORIZED TENSOR AUTODIFF ENGINE
// Each node = dense row-major matrix. No per-scalar nodes.
// Identical design to the RNN-Rust autograd — extended with ops needed for GPT:
//   • Reshape, Concat, Gather for attention
//   • SiLU activation for SwiGLU MLP
//   • RoPE (Rotary Position Embedding) graph op with analytic backward
//   • Causal-masked softmax attention (fused forward+backward)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TID(usize);

#[derive(Clone, Debug)]
enum TensorOp {
    None,
    // Arithmetic
    MatMul { a: TID, b: TID, m: usize, k: usize, n: usize },
    Add { a: TID, b: TID },
    Mul { a: TID, b: TID },
    Scale { a: TID, s: f32 },
    ScalarDiv { a: TID, n: f32 },
    Neg { a: TID },
    // Activations
    ReLU { a: TID },
    SiLU { a: TID },  // x * sigmoid(x)  — used by SwiGLU
    Exp { a: TID },
    Log { a: TID },
    // Normalization
    Square { a: TID },
    Sqrt { a: TID },
    Pow { a: TID, n: f32 },
    // Shape ops
    BroadcastAdd { a: TID, bias: TID, rows: usize, cols: usize },
    BroadcastMul { a: TID, scale: TID, rows: usize, cols: usize },
    ReduceMeanCols { a: TID, rows: usize, cols: usize },
    ExpandCol { a: TID, rows: usize, cols: usize },
    Transpose { a: TID, rows: usize, cols: usize },
    RowSlice { a: TID, row: usize, cols: usize },
    RowConcat { parts: Vec<TID>, each_cols: usize },
    /// Extract columns [col_start..col_start+out_cols] from [rows, total_cols] -> [rows, out_cols]
    ExtractCols { a: TID, rows: usize, total_cols: usize, col_start: usize, out_cols: usize },
    /// Concatenate column-wise: list of [rows, head_dim] -> [rows, n_heads*head_dim]
    ConcatCols { parts: Vec<TID>, rows: usize, each_cols: usize },
    SumAll { a: TID },
    // Loss
    SoftmaxCE { logits: TID, target_idx: usize, vocab: usize },
    /// Stack multiple [1, cols] rows into [n, cols]
    StackRows { parts: Vec<TID>, cols: usize },
    // Positional encoding
    /// Rotary Position Embedding: [seq_len, head_dim] -> [seq_len, head_dim]
    /// Rotates pairs (2i, 2i+1) by position-dependent angles.
    /// cos_sin stores precomputed [seq_len * half_dim] cos then [seq_len * half_dim] sin.
    RoPE { a: TID, seq_len: usize, head_dim: usize, cos_sin: Vec<f32> },
    // Attention (fused for efficiency)
    CausalAttnHead {
        q: TID, // [seq, head_dim]
        k: TID, // [seq, head_dim]
        v: TID, // [seq, head_dim]
        seq_len: usize,
        head_dim: usize,
    },
}

struct TensorNode {
    data: Vec<f32>,
    grad: Vec<f32>,
    rows: usize,
    cols: usize,
    op: TensorOp,
    is_param: bool,
}

struct Graph {
    nodes: Vec<TensorNode>,
    param_boundary: usize,
}

impl Graph {
    fn new() -> Self {
        Graph { nodes: Vec::with_capacity(8192), param_boundary: 0 }
    }

    fn param(&mut self, data: Vec<f32>, rows: usize, cols: usize) -> TID {
        let len = rows * cols;
        debug_assert_eq!(data.len(), len);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows, cols,
            op: TensorOp::None, is_param: true,
        });
        TID(id)
    }

    fn constant(&mut self, data: Vec<f32>, rows: usize, cols: usize) -> TID {
        let len = rows * cols;
        debug_assert_eq!(data.len(), len);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows, cols,
            op: TensorOp::None, is_param: false,
        });
        TID(id)
    }

    fn freeze_params(&mut self) { self.param_boundary = self.nodes.len(); }
    fn reset(&mut self) { self.nodes.truncate(self.param_boundary); }

    fn zero_grad(&mut self) {
        for node in self.nodes.iter_mut() {
            for g in node.grad.iter_mut() { *g = 0.0; }
        }
    }

    fn data(&self, t: TID) -> &[f32] { &self.nodes[t.0].data }
    fn rows(&self, t: TID) -> usize { self.nodes[t.0].rows }
    fn cols(&self, t: TID) -> usize { self.nodes[t.0].cols }

    // ------------------------------------------------------------------
    // Forward ops
    // ------------------------------------------------------------------

    /// Matrix multiply [m,k] @ [k,n] -> [m,n] via matrixmultiply::sgemm
    fn matmul(&mut self, a: TID, b: TID) -> TID {
        let m = self.nodes[a.0].rows;
        let k = self.nodes[a.0].cols;
        let n = self.nodes[b.0].cols;
        debug_assert_eq!(k, self.nodes[b.0].rows, "matmul shape [{},{}] @ [{},{}]",
            m, k, self.nodes[b.0].rows, n);

        let a_data = &self.nodes[a.0].data;
        let b_data = &self.nodes[b.0].data;
        let mut out = vec![0.0f32; m * n];

        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0,
                a_data.as_ptr(), k as isize, 1,
                b_data.as_ptr(), n as isize, 1,
                0.0,
                out.as_mut_ptr(), n as isize, 1,
            );
        }

        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; m * n], rows: m, cols: n,
            op: TensorOp::MatMul { a, b, m, k, n }, is_param: false,
        });
        TID(id)
    }

    fn add(&mut self, a: TID, b: TID) -> TID {
        let data = vec_add(&self.nodes[a.0].data, &self.nodes[b.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Add { a, b }, is_param: false,
        });
        TID(id)
    }

    fn mul(&mut self, a: TID, b: TID) -> TID {
        let data = vec_mul(&self.nodes[a.0].data, &self.nodes[b.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Mul { a, b }, is_param: false,
        });
        TID(id)
    }

    fn scale(&mut self, a: TID, s: f32) -> TID {
        let data = vec_scale(&self.nodes[a.0].data, s);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Scale { a, s }, is_param: false,
        });
        TID(id)
    }

    fn scalar_div(&mut self, a: TID, n: f32) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| x / n).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::ScalarDiv { a, n }, is_param: false,
        });
        TID(id)
    }

    fn neg(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| -x).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Neg { a }, is_param: false,
        });
        TID(id)
    }

    fn relu(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.max(0.0)).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::ReLU { a }, is_param: false,
        });
        TID(id)
    }

    /// SiLU (Swish): x * sigmoid(x)
    fn silu(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter()
            .map(|&x| {
                let s = 1.0 / (1.0 + (-x).exp()); // sigmoid(x)
                x * s
            }).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::SiLU { a }, is_param: false,
        });
        TID(id)
    }

    fn exp_op(&mut self, a: TID) -> TID {
        let data = vec_exp(&self.nodes[a.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Exp { a }, is_param: false,
        });
        TID(id)
    }

    fn log_op(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| (x + 1e-8).ln()).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Log { a }, is_param: false,
        });
        TID(id)
    }

    fn square(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| x * x).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Square { a }, is_param: false,
        });
        TID(id)
    }

    fn sqrt_op(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| (x + 1e-8).sqrt()).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Sqrt { a }, is_param: false,
        });
        TID(id)
    }

    fn pow_op(&mut self, a: TID, n: f32) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.powf(n)).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Pow { a, n }, is_param: false,
        });
        TID(id)
    }

    /// [m,n] + [1,n] broadcast bias add
    fn broadcast_add(&mut self, a: TID, bias: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let bd = &self.nodes[bias.0].data;
        let mut out = ad.clone();
        for i in 0..rows {
            let off = i * cols;
            for j in 0..cols { out[off + j] += bd[j]; }
        }
        let len = out.len();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; len], rows, cols,
            op: TensorOp::BroadcastAdd { a, bias, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// [m,n] * [1,n] broadcast element multiply
    fn broadcast_mul(&mut self, a: TID, sc: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let sd = &self.nodes[sc.0].data;
        let mut out = ad.clone();
        for i in 0..rows {
            let off = i * cols;
            for j in 0..cols { out[off + j] *= sd[j]; }
        }
        let len = out.len();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; len], rows, cols,
            op: TensorOp::BroadcastMul { a, scale: sc, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// Reduce mean along cols: [m,n] -> [m,1]
    fn reduce_mean_cols(&mut self, a: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let inv = 1.0 / cols as f32;
        let mut out = vec![0.0; rows];
        for i in 0..rows {
            let off = i * cols;
            let mut s = 0.0;
            for j in 0..cols { s += ad[off + j]; }
            out[i] = s * inv;
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows], rows, cols: 1,
            op: TensorOp::ReduceMeanCols { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// Expand [m,1] -> [m,n]
    fn expand_col(&mut self, a: TID, cols: usize) -> TID {
        let rows = self.nodes[a.0].rows;
        let ad = &self.nodes[a.0].data;
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            let v = ad[i];
            let off = i * cols;
            for j in 0..cols { out[off + j] = v; }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows * cols], rows, cols,
            op: TensorOp::ExpandCol { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    fn transpose(&mut self, a: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = ad[i * cols + j];
            }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows * cols], rows: cols, cols: rows,
            op: TensorOp::Transpose { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    fn row_slice(&mut self, a: TID, row: usize) -> TID {
        let cols = self.nodes[a.0].cols;
        let off = row * cols;
        let data = self.nodes[a.0].data[off..off + cols].to_vec();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; cols], rows: 1, cols,
            op: TensorOp::RowSlice { a, row, cols }, is_param: false,
        });
        TID(id)
    }

    /// Concatenate multiple [1, each_cols] row vectors into [1, sum_cols]
    fn row_concat(&mut self, parts: &[TID]) -> TID {
        let each_cols = self.nodes[parts[0].0].cols;
        let total_cols = each_cols * parts.len();
        let mut data = Vec::with_capacity(total_cols);
        for &p in parts {
            data.extend_from_slice(&self.nodes[p.0].data);
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; total_cols], rows: 1, cols: total_cols,
            op: TensorOp::RowConcat { parts: parts.to_vec(), each_cols }, is_param: false,
        });
        TID(id)
    }

    /// Apply Rotary Position Embedding to [seq_len, head_dim].
    /// cos_table/sin_table: precomputed [seq_len, half_dim] flattened.
    fn rope(&mut self, a: TID, cos_table: &[f32], sin_table: &[f32]) -> TID {
        let seq_len = self.nodes[a.0].rows;
        let head_dim = self.nodes[a.0].cols;
        let half = head_dim / 2;
        let src = &self.nodes[a.0].data;
        let mut out = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            let row = t * head_dim;
            let cs_off = t * half;
            for i in 0..half {
                let x0 = src[row + 2 * i];
                let x1 = src[row + 2 * i + 1];
                let c = cos_table[cs_off + i];
                let s = sin_table[cs_off + i];
                out[row + 2 * i]     = x0 * c - x1 * s;
                out[row + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
        // Pack cos then sin into one vec for storage in the op
        let mut cos_sin = Vec::with_capacity(cos_table.len() + sin_table.len());
        cos_sin.extend_from_slice(cos_table);
        cos_sin.extend_from_slice(sin_table);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; seq_len * head_dim], rows: seq_len, cols: head_dim,
            op: TensorOp::RoPE { a, seq_len, head_dim, cos_sin }, is_param: false,
        });
        TID(id)
    }

    /// Stack multiple [1, cols] row vectors into [n, cols] with gradient flow
    fn stack_rows(&mut self, parts: &[TID]) -> TID {
        let cols = self.nodes[parts[0].0].cols;
        let n = parts.len();
        let mut data = Vec::with_capacity(n * cols);
        for &p in parts {
            data.extend_from_slice(&self.nodes[p.0].data);
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; n * cols], rows: n, cols,
            op: TensorOp::StackRows { parts: parts.to_vec(), cols }, is_param: false,
        });
        TID(id)
    }

    /// Extract columns [col_start..col_start+out_cols] from [rows, total_cols] -> [rows, out_cols]
    fn extract_cols(&mut self, a: TID, col_start: usize, out_cols: usize) -> TID {
        let rows = self.nodes[a.0].rows;
        let total_cols = self.nodes[a.0].cols;
        let src = &self.nodes[a.0].data;
        let mut out = vec![0.0f32; rows * out_cols];
        for r in 0..rows {
            for d in 0..out_cols {
                out[r * out_cols + d] = src[r * total_cols + col_start + d];
            }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows * out_cols], rows, cols: out_cols,
            op: TensorOp::ExtractCols { a, rows, total_cols, col_start, out_cols }, is_param: false,
        });
        TID(id)
    }

    /// Concatenate column-wise: list of [rows, each_cols] -> [rows, n_parts*each_cols]
    fn concat_cols(&mut self, parts: &[TID]) -> TID {
        let rows = self.nodes[parts[0].0].rows;
        let each_cols = self.nodes[parts[0].0].cols;
        let n_parts = parts.len();
        let total_cols = n_parts * each_cols;
        let mut data = vec![0.0f32; rows * total_cols];
        for (p, &pid) in parts.iter().enumerate() {
            let p_data = &self.nodes[pid.0].data;
            for r in 0..rows {
                for d in 0..each_cols {
                    data[r * total_cols + p * each_cols + d] = p_data[r * each_cols + d];
                }
            }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; rows * total_cols], rows, cols: total_cols,
            op: TensorOp::ConcatCols { parts: parts.to_vec(), rows, each_cols }, is_param: false,
        });
        TID(id)
    }

    fn sum_all(&mut self, a: TID) -> TID {
        let s: f32 = self.nodes[a.0].data.iter().sum();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: vec![s], grad: vec![0.0], rows: 1, cols: 1,
            op: TensorOp::SumAll { a }, is_param: false,
        });
        TID(id)
    }

    /// Fused softmax cross-entropy: [1, vocab] + target -> scalar loss
    fn softmax_ce(&mut self, logits: TID, target_idx: usize) -> TID {
        let vocab = self.nodes[logits.0].cols;
        let ld = &self.nodes[logits.0].data;
        let max_l = vec_max(ld);
        let mut sum_exp = 0.0f32;
        for j in 0..vocab { sum_exp += (ld[j] - max_l).exp(); }
        let loss = sum_exp.ln() + max_l - ld[target_idx];
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: vec![loss], grad: vec![0.0], rows: 1, cols: 1,
            op: TensorOp::SoftmaxCE { logits, target_idx, vocab }, is_param: false,
        });
        TID(id)
    }

    /// Fused causal self-attention for one head.
    /// Q,K,V are [seq_len, head_dim]. Output is [seq_len, head_dim].
    /// Applies causal mask and scaled dot-product attention in one node.
    fn causal_attn_head(&mut self, q: TID, k: TID, v: TID) -> TID {
        let seq_len = self.nodes[q.0].rows;
        let head_dim = self.nodes[q.0].cols;
        debug_assert_eq!(self.nodes[k.0].rows, seq_len);
        debug_assert_eq!(self.nodes[v.0].rows, seq_len);
        debug_assert_eq!(self.nodes[k.0].cols, head_dim);
        debug_assert_eq!(self.nodes[v.0].cols, head_dim);

        let q_d = &self.nodes[q.0].data;
        let k_d = &self.nodes[k.0].data;
        let v_d = &self.nodes[v.0].data;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * head_dim];

        // For each query position t, attend to positions 0..=t
        for t in 0..seq_len {
            // Compute scaled attention scores for positions 0..=t
            let q_off = t * head_dim;
            let num_keys = t + 1;

            // scores[s] = sum_d Q[t,d] * K[s,d] * scale, for s in 0..=t
            let mut scores = vec![0.0f32; num_keys];
            for s in 0..num_keys {
                let k_off = s * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_d[q_off + d] * k_d[k_off + d];
                }
                scores[s] = dot * scale;
            }

            // Softmax over scores
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; num_keys];
            let mut sum_e = 0.0f32;
            for s in 0..num_keys {
                exps[s] = (scores[s] - max_s).exp();
                sum_e += exps[s];
            }
            let inv_sum = 1.0 / (sum_e + 1e-8);

            // attn_out[t, d] = sum_s attn_weight[s] * V[s, d]
            let out_off = t * head_dim;
            for s in 0..num_keys {
                let w = exps[s] * inv_sum;
                let v_off = s * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] += w * v_d[v_off + d];
                }
            }
        }

        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; seq_len * head_dim], rows: seq_len, cols: head_dim,
            op: TensorOp::CausalAttnHead { q, k, v, seq_len, head_dim }, is_param: false,
        });
        TID(id)
    }

    // ------------------------------------------------------------------
    // Backward — vectorized gradient propagation
    // ------------------------------------------------------------------

    fn backward(&mut self, loss: TID) {
        self.nodes[loss.0].grad = vec![1.0];
        let n = self.nodes.len();

        for i in (0..n).rev() {
            let has_grad = self.nodes[i].grad.iter().any(|&g| g != 0.0);
            if !has_grad { continue; }

            let op = self.nodes[i].op.clone();
            match op {
                TensorOp::None => {}

                TensorOp::MatMul { a, b, m, k, n: nn } => {
                    let og = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let b_d = self.nodes[b.0].data.clone();
                    // dA = dOut @ B^T  — stride-transposed, no copy
                    {
                        let mut da_buf = vec![0.0f32; m * k];
                        unsafe {
                            matrixmultiply::sgemm(
                                m, nn, k,
                                1.0,
                                og.as_ptr(), nn as isize, 1,
                                b_d.as_ptr(), 1, nn as isize, // B transposed via strides
                                0.0,
                                da_buf.as_mut_ptr(), k as isize, 1,
                            );
                        }
                        vec_add_inplace(&mut self.nodes[a.0].grad, &da_buf);
                    }
                    // dB = A^T @ dOut — stride-transposed, no copy
                    {
                        let mut db_buf = vec![0.0f32; k * nn];
                        unsafe {
                            matrixmultiply::sgemm(
                                k, m, nn,
                                1.0,
                                a_d.as_ptr(), 1, k as isize, // A transposed via strides
                                og.as_ptr(), nn as isize, 1,
                                0.0,
                                db_buf.as_mut_ptr(), nn as isize, 1,
                            );
                        }
                        vec_add_inplace(&mut self.nodes[b.0].grad, &db_buf);
                    }
                }

                TensorOp::Add { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                    vec_add_inplace(&mut self.nodes[b.0].grad, &g);
                }

                TensorOp::Mul { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let b_d = self.nodes[b.0].data.clone();
                    for j in 0..g.len() {
                        self.nodes[a.0].grad[j] += g[j] * b_d[j];
                        self.nodes[b.0].grad[j] += g[j] * a_d[j];
                    }
                }

                TensorOp::Scale { a, s } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * s; }
                }

                TensorOp::ScalarDiv { a, n: dv } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] / dv; }
                }

                TensorOp::Neg { a } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] -= g[j]; }
                }

                TensorOp::ReLU { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = &self.nodes[i].data;
                    for j in 0..g.len() {
                        if od[j] > 0.0 { self.nodes[a.0].grad[j] += g[j]; }
                    }
                }

                // SiLU backward: d/dx [x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
                //                                = σ(x) * (1 + x * (1 - σ(x)))
                TensorOp::SiLU { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() {
                        let x = a_d[j];
                        let sig = 1.0 / (1.0 + (-x).exp());
                        self.nodes[a.0].grad[j] += g[j] * sig * (1.0 + x * (1.0 - sig));
                    }
                }

                TensorOp::Exp { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * od[j]; }
                }

                TensorOp::Log { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] / (a_d[j] + 1e-8); }
                }

                TensorOp::Square { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * 2.0 * a_d[j]; }
                }

                TensorOp::Sqrt { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * 0.5 / od[j]; }
                }

                TensorOp::Pow { a, n: pw } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() {
                        self.nodes[a.0].grad[j] += g[j] * pw * a_d[j].powf(pw - 1.0);
                    }
                }

                TensorOp::BroadcastAdd { a, bias, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                    for ii in 0..rr {
                        let off = ii * cc;
                        for j in 0..cc { self.nodes[bias.0].grad[j] += g[off + j]; }
                    }
                }

                TensorOp::BroadcastMul { a, scale: sc, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let s_d = self.nodes[sc.0].data.clone();
                    for ii in 0..rr {
                        let off = ii * cc;
                        for j in 0..cc {
                            self.nodes[a.0].grad[off + j] += g[off + j] * s_d[j];
                            self.nodes[sc.0].grad[j] += g[off + j] * a_d[off + j];
                        }
                    }
                }

                TensorOp::ReduceMeanCols { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let inv = 1.0 / cc as f32;
                    for ii in 0..rr {
                        let off = ii * cc;
                        let gv = g[ii] * inv;
                        for j in 0..cc { self.nodes[a.0].grad[off + j] += gv; }
                    }
                }

                TensorOp::ExpandCol { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    for ii in 0..rr {
                        let off = ii * cc;
                        let mut s = 0.0;
                        for j in 0..cc { s += g[off + j]; }
                        self.nodes[a.0].grad[ii] += s;
                    }
                }

                TensorOp::Transpose { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    for ii in 0..cc {
                        for jj in 0..rr {
                            self.nodes[a.0].grad[jj * cc + ii] += g[ii * rr + jj];
                        }
                    }
                }

                TensorOp::RowSlice { a, row, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let off = row * cc;
                    for j in 0..cc { self.nodes[a.0].grad[off + j] += g[j]; }
                }

                TensorOp::RowConcat { parts, each_cols } => {
                    let g = self.nodes[i].grad.clone();
                    for (pi, &part_id) in parts.iter().enumerate() {
                        let off = pi * each_cols;
                        for j in 0..each_cols {
                            self.nodes[part_id.0].grad[j] += g[off + j];
                        }
                    }
                }

                TensorOp::StackRows { parts, cols: cc } => {
                    let og = self.nodes[i].grad.clone();
                    for (pi, &part_id) in parts.iter().enumerate() {
                        let off = pi * cc;
                        for j in 0..cc {
                            self.nodes[part_id.0].grad[j] += og[off + j];
                        }
                    }
                }

                TensorOp::ExtractCols { a, rows: rr, total_cols, col_start, out_cols } => {
                    let og = self.nodes[i].grad.clone();
                    for r in 0..rr {
                        for d in 0..out_cols {
                            self.nodes[a.0].grad[r * total_cols + col_start + d] += og[r * out_cols + d];
                        }
                    }
                }

                TensorOp::ConcatCols { parts, rows: rr, each_cols } => {
                    let og = self.nodes[i].grad.clone();
                    let total_cols = parts.len() * each_cols;
                    for (p, &pid) in parts.iter().enumerate() {
                        for r in 0..rr {
                            for d in 0..each_cols {
                                self.nodes[pid.0].grad[r * each_cols + d] += og[r * total_cols + p * each_cols + d];
                            }
                        }
                    }
                }

                TensorOp::SumAll { a } => {
                    let gv = self.nodes[i].grad[0];
                    for v in self.nodes[a.0].grad.iter_mut() { *v += gv; }
                }

                TensorOp::SoftmaxCE { logits, target_idx, vocab } => {
                    let gv = self.nodes[i].grad[0];
                    let ld = &self.nodes[logits.0].data;
                    let max_l = vec_max(ld);
                    let exps: Vec<f32> = ld.iter().map(|&x| (x - max_l).exp()).collect();
                    let sum_e: f32 = exps.iter().sum();
                    for j in 0..vocab {
                        let prob = exps[j] / sum_e;
                        let tg = if j == target_idx { prob - 1.0 } else { prob };
                        self.nodes[logits.0].grad[j] += gv * tg;
                    }
                }

                // RoPE backward: inverse rotation (negate sin)
                // d/da: rotate gradient by -θ (transpose of rotation matrix)
                TensorOp::RoPE { a, seq_len, head_dim, ref cos_sin } => {
                    let og = self.nodes[i].grad.clone();
                    let half = head_dim / 2;
                    let cs_len = seq_len * half;
                    let cos_t = &cos_sin[..cs_len];
                    let sin_t = &cos_sin[cs_len..];
                    for t in 0..seq_len {
                        let row = t * head_dim;
                        let cs_off = t * half;
                        for ii in 0..half {
                            let g0 = og[row + 2 * ii];
                            let g1 = og[row + 2 * ii + 1];
                            let c = cos_t[cs_off + ii];
                            let s = sin_t[cs_off + ii];
                            // Inverse rotation: transpose of [c -s; s c] is [c s; -s c]
                            self.nodes[a.0].grad[row + 2 * ii]     += g0 * c + g1 * s;
                            self.nodes[a.0].grad[row + 2 * ii + 1] += -g0 * s + g1 * c;
                        }
                    }
                }

                TensorOp::CausalAttnHead { q, k, v, seq_len, head_dim } => {
                    let og = self.nodes[i].grad.clone();
                    let q_d = self.nodes[q.0].data.clone();
                    let k_d = self.nodes[k.0].data.clone();
                    let v_d = self.nodes[v.0].data.clone();

                    let scale = 1.0 / (head_dim as f32).sqrt();

                    for t in 0..seq_len {
                        let q_off = t * head_dim;
                        let num_keys = t + 1;

                        // Recompute softmax for this position
                        let mut scores = vec![0.0f32; num_keys];
                        for s in 0..num_keys {
                            let k_off = s * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim { dot += q_d[q_off + d] * k_d[k_off + d]; }
                            scores[s] = dot * scale;
                        }
                        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exps = vec![0.0f32; num_keys];
                        let mut sum_e = 0.0f32;
                        for s in 0..num_keys {
                            exps[s] = (scores[s] - max_s).exp();
                            sum_e += exps[s];
                        }
                        let inv_sum = 1.0 / (sum_e + 1e-8);
                        let weights: Vec<f32> = exps.iter().map(|&e| e * inv_sum).collect();

                        // d_out for this position t
                        let out_off = t * head_dim;

                        // d_weights[s] = sum_d dOut[t,d] * V[s,d]
                        let mut d_weights = vec![0.0f32; num_keys];
                        for s in 0..num_keys {
                            let v_off = s * head_dim;
                            let mut dw = 0.0f32;
                            for d in 0..head_dim {
                                dw += og[out_off + d] * v_d[v_off + d];
                            }
                            d_weights[s] = dw;
                        }

                        // Backprop through softmax: d_scores = w * (d_weights - sum(w * d_weights))
                        let dot_wd: f32 = weights.iter().zip(d_weights.iter()).map(|(w, dw)| w * dw).sum();
                        let mut d_scores = vec![0.0f32; num_keys];
                        for s in 0..num_keys {
                            d_scores[s] = weights[s] * (d_weights[s] - dot_wd);
                        }

                        // Grad for V: dV[s,d] += weights[s] * dOut[t,d]
                        for s in 0..num_keys {
                            let v_off = s * head_dim;
                            let w = weights[s];
                            for d in 0..head_dim {
                                self.nodes[v.0].grad[v_off + d] += w * og[out_off + d];
                            }
                        }

                        // Grad for Q,K from d_scores (through scaled dot product)
                        for s in 0..num_keys {
                            let k_off = s * head_dim;
                            let ds = d_scores[s] * scale;
                            for d in 0..head_dim {
                                // dQ[t,d] += d_score[s] * K[s,d] * scale
                                self.nodes[q.0].grad[q_off + d] += ds * k_d[k_off + d];
                                // dK[s,d] += d_score[s] * Q[t,d] * scale
                                self.nodes[k.0].grad[k_off + d] += ds * q_d[q_off + d];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// PARAMETER COLLECTION
// ============================================================================

struct ParamSet {
    ids: Vec<TID>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl ParamSet {
    fn new(ids: Vec<TID>, g: &Graph) -> Self {
        let m: Vec<Vec<f32>> = ids.iter().map(|&t| vec![0.0; g.nodes[t.0].data.len()]).collect();
        let v: Vec<Vec<f32>> = ids.iter().map(|&t| vec![0.0; g.nodes[t.0].data.len()]).collect();
        ParamSet { ids, m, v }
    }
}

// ============================================================================
// ADAM OPTIMIZER — matches the Python implementation
// ============================================================================

struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
}

impl Adam {
    fn new(lr: f32, beta1: f32, beta2: f32, wd: f32) -> Self {
        Adam { lr, beta1, beta2, eps: 1e-8, weight_decay: wd, t: 0 }
    }

    fn step(&mut self, g: &mut Graph, ps: &mut ParamSet, lr_override: Option<f32>) {
        self.t += 1;
        let lr = lr_override.unwrap_or(self.lr);
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (idx, &tid) in ps.ids.iter().enumerate() {
            let node = &mut g.nodes[tid.0];
            let len = node.data.len();
            let pm = &mut ps.m[idx];
            let pv = &mut ps.v[idx];
            for j in 0..len {
                let grad = node.grad[j];
                if grad == 0.0 && pm[j] == 0.0 { continue; } // sparse skip
                pm[j] = self.beta1 * pm[j] + (1.0 - self.beta1) * grad;
                pv[j] = self.beta2 * pv[j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = pm[j] / bc1;
                let v_hat = pv[j] / bc2;
                if self.weight_decay > 0.0 {
                    node.data[j] -= lr * self.weight_decay * node.data[j];
                }
                node.data[j] -= lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

// ============================================================================
// RMS NORM
// ============================================================================

/// RMSNorm: x * (mean(x^2) + eps)^{-0.5}
/// No learned weight — matches the Python GPT's rmsnorm exactly.
fn rmsnorm(x: TID, g: &mut Graph) -> TID {
    let cols = g.cols(x);
    let sq = g.square(x);
    let mean_sq = g.reduce_mean_cols(sq);         // [rows, 1]
    let eps_c = g.constant(vec![1e-5; g.rows(mean_sq)], g.rows(mean_sq), 1);
    let mean_sq_eps = g.add(mean_sq, eps_c);       // [rows, 1]
    let inv_rms = g.pow_op(mean_sq_eps, -0.5);     // [rows, 1]
    let inv_rms_exp = g.expand_col(inv_rms, cols); // [rows, cols]
    g.mul(x, inv_rms_exp)
}

// ============================================================================
// GPT MODEL
// ============================================================================

struct GPT {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    n_layer: usize,
    block_size: usize,
    head_dim: usize,
    kv_dim: usize,     // n_kv_head * head_dim (total KV projection width)
    group_size: usize,  // n_head / n_kv_head (query heads per KV head)
    n_experts: usize,   // total experts (expert 0 = shared always-on)
    top_k: usize,       // sparse experts activated per token
    // Embeddings (RoPE replaces learned positional embeddings)
    wte: TID,  // [vocab_size, n_embd]
    // Precomputed RoPE cos/sin tables: [block_size, head_dim/2] flattened
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    // Per-layer parameters  (GQA: K,V are smaller than Q)
    attn_wq: Vec<TID>, // [n_embd, n_embd]       — full Q projection
    attn_wk: Vec<TID>, // [n_embd, kv_dim]       — reduced K projection
    attn_wv: Vec<TID>, // [n_embd, kv_dim]       — reduced V projection
    attn_wo: Vec<TID>, // [n_embd, n_embd]       — output projection
    // MoE: expert 0 = always-on shared expert, experts 1..n_experts-1 = sparse
    // Each expert has 3 SwiGLU weight matrices:
    moe_fc1:  Vec<Vec<TID>>,  // [n_layer][n_experts] each [n_embd, 4*n_embd]
    moe_gate: Vec<Vec<TID>>,  // [n_layer][n_experts] each [n_embd, 4*n_embd]
    moe_fc2:  Vec<Vec<TID>>,  // [n_layer][n_experts] each [4*n_embd, n_embd]
    moe_router: Vec<TID>,     // [n_layer] each [n_embd, n_experts-1]  (routes sparse experts)
    // Output head
    lm_head: TID, // [n_embd, vocab_size]
}

impl GPT {
    fn new(
        vocab_size: usize, n_embd: usize, n_head: usize, n_kv_head: usize,
        n_layer: usize, block_size: usize, n_experts: usize, top_k: usize,
        g: &mut Graph, rng: &mut impl Rng,
    ) -> Self {
        assert!(n_head % n_kv_head == 0,
            "n_head ({}) must be divisible by n_kv_head ({})", n_head, n_kv_head);
        assert!(n_embd % n_head == 0,
            "n_embd ({}) must be divisible by n_head ({})", n_embd, n_head);
        assert!((n_embd / n_head) % 2 == 0,
            "head_dim ({}) must be even for RoPE", n_embd / n_head);
        assert!(n_experts >= 2,
            "n_experts ({}) must be >= 2 (1 shared + at least 1 sparse)", n_experts);
        assert!(top_k <= n_experts - 1,
            "top_k ({}) must be <= n_experts-1 ({})", top_k, n_experts - 1);

        let head_dim = n_embd / n_head;
        let kv_dim = n_kv_head * head_dim;  // total width of K,V projections
        let group_size = n_head / n_kv_head; // query heads sharing each KV head
        let n_sparse = n_experts - 1;
        let std_init = 0.08;

        let wte = g.param(rand_normal(vocab_size * n_embd, std_init, rng), vocab_size, n_embd);

        // Precompute RoPE cos/sin tables: θ_i = 1 / 10000^(2i/head_dim)
        let half = head_dim / 2;
        let mut rope_cos = vec![0.0f32; block_size * half];
        let mut rope_sin = vec![0.0f32; block_size * half];
        for pos in 0..block_size {
            let off = pos * half;
            for i in 0..half {
                let theta = (pos as f32) / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                rope_cos[off + i] = theta.cos();
                rope_sin[off + i] = theta.sin();
            }
        }

        let mut attn_wq = Vec::with_capacity(n_layer);
        let mut attn_wk = Vec::with_capacity(n_layer);
        let mut attn_wv = Vec::with_capacity(n_layer);
        let mut attn_wo = Vec::with_capacity(n_layer);
        let mut moe_fc1  = Vec::with_capacity(n_layer);
        let mut moe_gate = Vec::with_capacity(n_layer);
        let mut moe_fc2  = Vec::with_capacity(n_layer);
        let mut moe_router = Vec::with_capacity(n_layer);

        for _ in 0..n_layer {
            // GQA: Q is full [n_embd, n_embd], K/V are reduced [n_embd, kv_dim]
            attn_wq.push(g.param(rand_normal(n_embd * n_embd, std_init, rng), n_embd, n_embd));
            attn_wk.push(g.param(rand_normal(n_embd * kv_dim, std_init, rng), n_embd, kv_dim));
            attn_wv.push(g.param(rand_normal(n_embd * kv_dim, std_init, rng), n_embd, kv_dim));
            attn_wo.push(g.param(rand_normal(n_embd * n_embd, std_init, rng), n_embd, n_embd));

            // MoE: n_experts SwiGLU experts + router for sparse experts
            let mut fc1_layer  = Vec::with_capacity(n_experts);
            let mut gate_layer = Vec::with_capacity(n_experts);
            let mut fc2_layer  = Vec::with_capacity(n_experts);
            for _ in 0..n_experts {
                fc1_layer.push(g.param(rand_normal(n_embd * 4 * n_embd, std_init, rng), n_embd, 4 * n_embd));
                gate_layer.push(g.param(rand_normal(n_embd * 4 * n_embd, std_init, rng), n_embd, 4 * n_embd));
                fc2_layer.push(g.param(rand_normal(4 * n_embd * n_embd, std_init, rng), 4 * n_embd, n_embd));
            }
            moe_fc1.push(fc1_layer);
            moe_gate.push(gate_layer);
            moe_fc2.push(fc2_layer);
            // Router: [n_embd, n_sparse] — only routes over sparse experts (not the shared one)
            moe_router.push(g.param(rand_normal(n_embd * n_sparse, std_init * 0.5, rng), n_embd, n_sparse));
        }

        let lm_head = g.param(rand_normal(n_embd * vocab_size, std_init, rng), n_embd, vocab_size);

        GPT {
            vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size,
            head_dim, kv_dim, group_size, n_experts, top_k,
            wte, rope_cos, rope_sin,
            attn_wq, attn_wk, attn_wv, attn_wo,
            moe_fc1, moe_gate, moe_fc2, moe_router, lm_head,
        }
    }

    fn param_ids(&self) -> Vec<TID> {
        let mut ids = vec![self.wte];
        for i in 0..self.n_layer {
            ids.extend([
                self.attn_wq[i], self.attn_wk[i], self.attn_wv[i], self.attn_wo[i],
            ]);
            // MoE: router + all expert weights
            ids.push(self.moe_router[i]);
            for e in 0..self.n_experts {
                ids.push(self.moe_fc1[i][e]);
                ids.push(self.moe_gate[i][e]);
                ids.push(self.moe_fc2[i][e]);
            }
        }
        ids.push(self.lm_head);
        ids
    }

    fn total_params(&self, g: &Graph) -> usize {
        self.param_ids().iter().map(|&t| g.nodes[t.0].data.len()).sum()
    }

    /// Full-sequence forward pass.
    /// tokens: &[usize] of length seq_len (≤ block_size).
    /// Returns logits TIDs: one [1, vocab_size] per position.
    fn forward(&self, tokens: &[usize], g: &mut Graph) -> Vec<TID> {
        let seq_len = tokens.len();
        assert!(seq_len <= self.block_size);

        // 1. Embed tokens into [seq_len, n_embd] (RoPE handles position — no wpe)
        let mut embed_rows: Vec<TID> = Vec::with_capacity(seq_len);
        for &tok in tokens.iter() {
            let tok_emb = g.row_slice(self.wte, tok); // [1, n_embd]
            embed_rows.push(tok_emb);
        }

        // Stack into [seq_len, n_embd] — gradient flows back to wte via StackRows op
        let x_stacked = self.stack_rows_op(g, &embed_rows); // [seq_len, n_embd]

        // Initial RMSNorm
        let mut x = rmsnorm(x_stacked, g); // [seq_len, n_embd]

        for li in 0..self.n_layer {
            // ============ Multi-Head Attention ============
            let x_residual = x;
            x = rmsnorm(x, g);

            // === Grouped Query Attention (GQA) ===
            // Q: full width [seq_len, n_embd], K/V: reduced [seq_len, kv_dim]
            let q_all = g.matmul(x, self.attn_wq[li]); // [seq_len, n_embd]
            let k_all = g.matmul(x, self.attn_wk[li]); // [seq_len, kv_dim]
            let v_all = g.matmul(x, self.attn_wv[li]); // [seq_len, kv_dim]

            let hd = self.head_dim;
            let ne = self.n_embd;
            let kv_dim = self.kv_dim;
            let group_size = self.group_size;

            // Precompute RoPE cos/sin slice for this sequence length
            let half = hd / 2;
            let rope_cos: Vec<f32> = self.rope_cos[..seq_len * half].to_vec();
            let rope_sin: Vec<f32> = self.rope_sin[..seq_len * half].to_vec();

            // ---- Parallel GQA via Rayon ----
            // Each of n_head query heads runs in parallel.
            // Query head h uses KV head (h / group_size).
            let q_data = g.data(q_all).to_vec();
            let k_data = g.data(k_all).to_vec();
            let v_data = g.data(v_all).to_vec();

            let head_indices: Vec<usize> = (0..self.n_head).collect();
            let _head_outputs: Vec<Vec<f32>> = head_indices.par_iter().map(|&h| {
                let q_start = h * hd;              // Q columns for this head
                let kv_h = h / group_size;          // which KV head to use
                let kv_start = kv_h * hd;           // KV columns for the shared head

                // Extract Q [seq_len, head_dim] from q_data [seq_len, n_embd]
                let mut q_h = vec![0.0f32; seq_len * hd];
                for t in 0..seq_len {
                    for d in 0..hd { q_h[t * hd + d] = q_data[t * ne + q_start + d]; }
                }
                // Extract K,V [seq_len, head_dim] from k/v_data [seq_len, kv_dim]
                let mut k_h = vec![0.0f32; seq_len * hd];
                let mut v_h = vec![0.0f32; seq_len * hd];
                for t in 0..seq_len {
                    for d in 0..hd {
                        k_h[t * hd + d] = k_data[t * kv_dim + kv_start + d];
                        v_h[t * hd + d] = v_data[t * kv_dim + kv_start + d];
                    }
                }

                // Apply RoPE to Q and K (pure computation)
                for t in 0..seq_len {
                    let row = t * hd;
                    let cs_off = t * half;
                    for ii in 0..half {
                        let c = rope_cos[cs_off + ii];
                        let s = rope_sin[cs_off + ii];
                        let q0 = q_h[row + 2 * ii];
                        let q1 = q_h[row + 2 * ii + 1];
                        q_h[row + 2 * ii]     = q0 * c - q1 * s;
                        q_h[row + 2 * ii + 1] = q0 * s + q1 * c;
                        let k0 = k_h[row + 2 * ii];
                        let k1 = k_h[row + 2 * ii + 1];
                        k_h[row + 2 * ii]     = k0 * c - k1 * s;
                        k_h[row + 2 * ii + 1] = k0 * s + k1 * c;
                    }
                }

                // Causal attention (pure computation for speed)
                let scale = 1.0 / (hd as f32).sqrt();
                let mut out = vec![0.0f32; seq_len * hd];
                for t in 0..seq_len {
                    let q_off = t * hd;
                    let num_keys = t + 1;
                    let mut scores = vec![0.0f32; num_keys];
                    for s in 0..num_keys {
                        let k_off = s * hd;
                        let mut dot = 0.0f32;
                        for di in 0..hd { dot += q_h[q_off + di] * k_h[k_off + di]; }
                        scores[s] = dot * scale;
                    }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps_s = vec![0.0f32; num_keys];
                    let mut sum_e = 0.0f32;
                    for s in 0..num_keys { exps_s[s] = (scores[s] - max_s).exp(); sum_e += exps_s[s]; }
                    let inv_sum = 1.0 / (sum_e + 1e-8);
                    let out_off = t * hd;
                    for s in 0..num_keys {
                        let w = exps_s[s] * inv_sum;
                        let v_off = s * hd;
                        for di in 0..hd { out[out_off + di] += w * v_h[v_off + di]; }
                    }
                }
                out
            }).collect();

            // Replay through graph for backprop (GQA: multiple Q heads share same K/V slice)
            // Pre-extract each KV head's slice once, then reuse for all Q heads in the group
            let mut kv_head_slices: Vec<(TID, TID)> = Vec::with_capacity(self.n_kv_head);
            for kv_h in 0..self.n_kv_head {
                let kv_start = kv_h * hd;
                let k_h = g.extract_cols(k_all, kv_start, hd); // [seq_len, head_dim]
                let v_h = g.extract_cols(v_all, kv_start, hd);
                kv_head_slices.push((k_h, v_h));
            }

            // Apply RoPE to each KV head's K slice (once per KV head)
            let rope_cos_sl = &self.rope_cos[..seq_len * half];
            let rope_sin_sl = &self.rope_sin[..seq_len * half];
            let kv_head_slices: Vec<(TID, TID)> = kv_head_slices.into_iter()
                .map(|(k_h, v_h)| {
                    let k_h_rope = g.rope(k_h, rope_cos_sl, rope_sin_sl);
                    (k_h_rope, v_h)  // RoPE on K only, V unchanged
                }).collect();

            let mut attn_head_results: Vec<TID> = Vec::with_capacity(self.n_head);
            for h in 0..self.n_head {
                let q_start = h * hd;
                let kv_h = h / group_size;
                let q_h = g.extract_cols(q_all, q_start, hd);     // [seq_len, head_dim]
                let q_h_rope = g.rope(q_h, rope_cos_sl, rope_sin_sl); // RoPE on Q
                let (k_h, v_h) = kv_head_slices[kv_h];            // shared KV head (already RoPE'd)
                let attn_out = g.causal_attn_head(q_h_rope, k_h, v_h); // fused causal attention
                attn_head_results.push(attn_out);
            }

            // Concatenate heads: n_head * [seq_len, head_dim] -> [seq_len, n_embd]
            let attn_cat = self.concat_heads(g, &attn_head_results, seq_len, hd);

            // Output projection
            let attn_proj = g.matmul(attn_cat, self.attn_wo[li]); // [seq_len, n_embd]

            // Residual connection
            x = g.add(attn_proj, x_residual);

            // ============ Mixture of Experts (MoE) Block ============
            // Expert 0: always-on shared expert (runs on full sequence)
            // Experts 1..n_experts-1: sparse, activated per-token via router top-k
            //
            // output = shared_expert(x) + Σ_{top_k} weight_i * expert_i(x)
            let x_residual2 = x;
            x = rmsnorm(x, g);

            // --- Shared expert 0: full-sequence SwiGLU (always on, weight=1) ---
            let sh_h1 = g.matmul(x, self.moe_fc1[li][0]);
            let sh_act = g.silu(sh_h1);
            let sh_gate = g.matmul(x, self.moe_gate[li][0]);
            let sh_gated = g.mul(sh_act, sh_gate);
            let shared_out = g.matmul(sh_gated, self.moe_fc2[li][0]); // [seq_len, n_embd]

            // --- Router: compute logits for sparse experts ---
            // x [seq_len, n_embd] @ router [n_embd, n_sparse] -> [seq_len, n_sparse]
            let n_sparse = self.n_experts - 1;
            let router_logits_tid = g.matmul(x, self.moe_router[li]);
            let router_data = g.data(router_logits_tid).to_vec();

            // Per-token routing: softmax over sparse experts, pick top_k
            let mut token_expert_selections: Vec<Vec<(usize, f32)>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let off = t * n_sparse;
                let logits_t = &router_data[off..off + n_sparse];
                // Softmax
                let max_l = logits_t.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits_t.iter().map(|&l| (l - max_l).exp()).collect();
                let sum_e: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|&e| e / (sum_e + 1e-8)).collect();
                // Top-k selection
                let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(self.top_k);
                // Renormalize selected weights
                let sel_sum: f32 = indexed.iter().map(|&(_, w)| w).sum();
                let inv = 1.0 / (sel_sum + 1e-8);
                let selected: Vec<(usize, f32)> = indexed.iter().map(|&(idx, w)| (idx, w * inv)).collect();
                token_expert_selections.push(selected);
            }

            // --- Build per-token expert outputs in the graph ---
            // For each token position:
            //   1. slice x_t = [1, n_embd]
            //   2. run each selected sparse expert on x_t (SwiGLU)
            //   3. weight by router prob, sum into sparse contribution
            //   4. add shared expert result for this token
            let mut moe_row_tids: Vec<TID> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = g.row_slice(x, t); // [1, n_embd]
                // Start with shared expert's output for this token
                let shared_t = g.row_slice(shared_out, t); // [1, n_embd]
                let mut combined = shared_t;

                for &(sparse_idx, weight) in &token_expert_selections[t] {
                    let expert_id = sparse_idx + 1; // sparse_idx 0..n_sparse maps to expert 1..n_experts
                    // SwiGLU for this expert on x_t
                    let e_h1 = g.matmul(x_t, self.moe_fc1[li][expert_id]);
                    let e_act = g.silu(e_h1);
                    let e_gate = g.matmul(x_t, self.moe_gate[li][expert_id]);
                    let e_gated = g.mul(e_act, e_gate);
                    let e_out = g.matmul(e_gated, self.moe_fc2[li][expert_id]); // [1, n_embd]
                    // Scale by router weight
                    let e_scaled = g.scale(e_out, weight);
                    combined = g.add(combined, e_scaled);
                }
                moe_row_tids.push(combined);
            }
            // Stack token rows back: [seq_len, n_embd]
            let moe_out = g.stack_rows(&moe_row_tids);
            x = g.add(moe_out, x_residual2);
        }

        // Output logits: one per position
        // x is [seq_len, n_embd], lm_head is [n_embd, vocab_size]
        let logits_all = g.matmul(x, self.lm_head); // [seq_len, vocab_size]

        // Split into per-position row slices
        let mut logits = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            logits.push(g.row_slice(logits_all, t));
        }
        logits
    }

    /// Stack a list of [1, cols] row vectors into [n, cols] — gradient flows back
    fn stack_rows_op(&self, g: &mut Graph, rows: &[TID]) -> TID {
        g.stack_rows(rows)
    }

    /// Concatenate head outputs [seq_len, head_dim] * n_head -> [seq_len, n_embd]
    /// Uses a graph op so gradients flow back through each head's attention output.
    fn concat_heads(&self, g: &mut Graph, heads: &[TID], _seq_len: usize, _head_dim: usize) -> TID {
        g.concat_cols(heads)
    }

    fn cross_entropy_loss(&self, logits: &[TID], targets: &[usize], g: &mut Graph) -> TID {
        let n = logits.len();
        let mut total = g.softmax_ce(logits[0], targets[0]);
        for i in 1..n {
            let ce = g.softmax_ce(logits[i], targets[i]);
            total = g.add(total, ce);
        }
        g.scalar_div(total, n as f32)
    }

    fn clip_grad(g: &mut Graph, pids: &[TID], max_norm: f32) -> f32 {
        let mut tn = 0.0f32;
        for &t in pids { for &gv in &g.nodes[t.0].grad { tn += gv * gv; } }
        tn = tn.sqrt();
        if tn > max_norm {
            let s = max_norm / tn;
            for &t in pids { vec_scale_inplace(&mut g.nodes[t.0].grad, s); }
        }
        tn
    }

    // ------------------------------------------------------------------
    // Checkpoint save / load
    // ------------------------------------------------------------------

    fn save_checkpoint(&self, fp: &str, meta: serde_json::Value, step: usize, g: &Graph, ps: &ParamSet) {
        let total_len: usize = ps.ids.iter().map(|&t| g.nodes[t.0].data.len()).sum();
        let mut data_vec = Vec::with_capacity(total_len);
        let mut m_vec = Vec::with_capacity(total_len);
        let mut v_vec = Vec::with_capacity(total_len);
        for (idx, &tid) in ps.ids.iter().enumerate() {
            data_vec.extend_from_slice(&g.nodes[tid.0].data);
            m_vec.extend_from_slice(&ps.m[idx]);
            v_vec.extend_from_slice(&ps.v[idx]);
        }
        let cp = Checkpoint {
            version: "1.0-gpt".to_string(),
            metadata: Some(serde_json::to_string(&meta).unwrap_or_default()),
            optimizer_step: step,
            config: CheckpointConfig {
                vocab_size: self.vocab_size, n_embd: self.n_embd,
                n_head: self.n_head, n_kv_head: self.n_kv_head,
                n_layer: self.n_layer, block_size: self.block_size,
                n_experts: self.n_experts, top_k: self.top_k,
            },
            param_data: data_vec, param_m: m_vec, param_v: v_vec,
        };
        let bytes = bincode::serialize(&cp).unwrap();
        fs::write(fp, &bytes).unwrap();
        println!("Saved checkpoint to {} ({:.2} MB)", fp, bytes.len() as f64 / 1_048_576.0);
    }

    fn load_checkpoint(&self, fp: &str, g: &mut Graph, ps: &mut ParamSet) -> Checkpoint {
        let bytes = fs::read(fp).unwrap();
        let cp: Checkpoint = bincode::deserialize(&bytes).unwrap();
        let mut off = 0;
        for (idx, &tid) in ps.ids.iter().enumerate() {
            let len = g.nodes[tid.0].data.len();
            g.nodes[tid.0].data[..len].copy_from_slice(&cp.param_data[off..off + len]);
            ps.m[idx][..len].copy_from_slice(&cp.param_m[off..off + len]);
            ps.v[idx][..len].copy_from_slice(&cp.param_v[off..off + len]);
            off += len;
        }
        println!("Loaded checkpoint from {} ({} params, {:.2} MB)", fp, off, bytes.len() as f64 / 1_048_576.0);
        cp
    }
}

// ============================================================================
// CHECKPOINT SERIALIZATION
// ============================================================================

#[derive(Serialize, Deserialize, Clone)]
struct Checkpoint {
    version: String,
    metadata: Option<String>,
    optimizer_step: usize,
    config: CheckpointConfig,
    param_data: Vec<f32>,
    param_m: Vec<f32>,
    param_v: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CheckpointConfig {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    n_layer: usize,
    block_size: usize,
    n_experts: usize,
    top_k: usize,
}

// ============================================================================
// UTILITIES
// ============================================================================

fn rand_normal(n: usize, scale: f32, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale
    }).collect()
}

// ============================================================================
// BPE TOKENIZER — train separately, save/load as JSON, use for model I/O
// ============================================================================

/// A Byte Pair Encoding tokenizer.
/// - Base vocab = all unique bytes (0..255 that appear) + special tokens
/// - Merges learned greedily from corpus
/// - Serialized to/from JSON for reuse
#[derive(Serialize, Deserialize, Clone)]
struct BpeTokenizer {
    /// Ordered merge rules: (pair_a, pair_b) -> merged_id
    merges: Vec<(usize, usize, usize)>,
    /// token_id -> byte sequence (the "word" for each token)
    vocab: Vec<Vec<u8>>,
    /// Special token ids
    bos_id: usize,
    eos_id: usize,
    /// For fast lookup: byte-seq -> token_id
    #[serde(skip)]
    piece_to_id: HashMap<Vec<u8>, usize>,
    /// For fast merge lookup: (a, b) -> merged_id
    #[serde(skip)]
    merge_map: HashMap<(usize, usize), usize>,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer from raw text.
    /// `target_vocab` is the desired total vocabulary size (including BOS/EOS).
    fn train(text: &str, target_vocab: usize) -> Self {
        println!("Training BPE tokenizer (target vocab = {})...", target_vocab);
        let t0 = Instant::now();

        // 1. Build base vocab from all unique bytes in the corpus
        let raw_bytes = text.as_bytes();
        let mut byte_set = [false; 256];
        for &b in raw_bytes { byte_set[b as usize] = true; }

        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut byte_to_id: [usize; 256] = [0; 256];
        for b in 0u16..256 {
            if byte_set[b as usize] {
                byte_to_id[b as usize] = vocab.len();
                vocab.push(vec![b as u8]);
            }
        }
        let base_vocab_size = vocab.len();
        // Add special tokens
        let bos_id = vocab.len();
        vocab.push(b"<BOS>".to_vec());
        let eos_id = vocab.len();
        vocab.push(b"<EOS>".to_vec());

        println!("  base vocab (unique bytes): {} + 2 special = {}",
            base_vocab_size, vocab.len());

        // 2. Tokenize the whole corpus at byte level, split by lines (documents)
        //    Each document is: [byte_id, byte_id, ...]
        let docs: Vec<Vec<usize>> = text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(|line| line.bytes().map(|b| byte_to_id[b as usize]).collect())
            .collect();

        // Working copy of all documents as token sequences
        let mut work: Vec<Vec<usize>> = docs;
        let mut merges: Vec<(usize, usize, usize)> = Vec::new();
        let n_merges = target_vocab.saturating_sub(vocab.len());

        println!("  planning {} merges...", n_merges);

        for mi in 0..n_merges {
            // 3. Count all adjacent pairs across all documents
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for doc in &work {
                for w in doc.windows(2) {
                    *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }

            // 4. Find the most frequent pair
            let best = pair_counts.into_iter()
                .max_by_key(|&(_, count)| count);

            let ((a, b), count) = match best {
                Some(x) if x.1 >= 2 => x,
                _ => { println!("  stopped early at merge {} (no pair with count >= 2)", mi); break; }
            };

            // 5. Create new merged token
            let new_id = vocab.len();
            let mut merged_bytes = vocab[a].clone();
            merged_bytes.extend_from_slice(&vocab[b]);
            vocab.push(merged_bytes);
            merges.push((a, b, new_id));

            // 6. Apply this merge to all documents
            for doc in work.iter_mut() {
                let mut i = 0;
                while i + 1 < doc.len() {
                    if doc[i] == a && doc[i + 1] == b {
                        doc[i] = new_id;
                        doc.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            if (mi + 1) % 50 == 0 || mi == 0 {
                println!("  merge {:4}/{}: ({:3},{:3}) -> {:3}  count={:6}  vocab={}",
                    mi + 1, n_merges, a, b, new_id, count, vocab.len());
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        println!("BPE training done in {:.1}s — final vocab size: {}", elapsed, vocab.len());

        let mut tok = BpeTokenizer {
            merges,
            vocab,
            bos_id,
            eos_id,
            piece_to_id: HashMap::new(),
            merge_map: HashMap::new(),
        };
        tok.rebuild_indexes();
        tok
    }

    /// Rebuild lookup indexes from merges/vocab (needed after deserialization).
    fn rebuild_indexes(&mut self) {
        self.piece_to_id.clear();
        for (id, piece) in self.vocab.iter().enumerate() {
            self.piece_to_id.insert(piece.clone(), id);
        }
        self.merge_map.clear();
        for &(a, b, new_id) in &self.merges {
            self.merge_map.insert((a, b), new_id);
        }
    }

    /// Encode a string into a sequence of token IDs.
    fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() { return vec![]; }
        // Start with byte-level token ids
        let mut ids: Vec<usize> = text.bytes().map(|b| {
            *self.piece_to_id.get(&vec![b]).unwrap_or(&0)
        }).collect();

        // Apply merges in order (greedy — same order as training)
        for &(a, b, new_id) in &self.merges {
            let mut i = 0;
            while i + 1 < ids.len() {
                if ids[i] == a && ids[i + 1] == b {
                    ids[i] = new_id;
                    ids.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        ids
    }

    /// Decode a sequence of token IDs back to a string.
    fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id == self.bos_id || id == self.eos_id { continue; }
            if id < self.vocab.len() {
                bytes.extend_from_slice(&self.vocab[id]);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Total vocabulary size (including special tokens).
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Save tokenizer to a JSON file.
    fn save(&self, path: &str) {
        let json = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, &json).unwrap();
        println!("Tokenizer saved to {} ({} tokens, {:.1} KB)",
            path, self.vocab.len(), json.len() as f64 / 1024.0);
    }

    /// Load tokenizer from a JSON file.
    fn load(path: &str) -> Self {
        let json = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Cannot read tokenizer file: {}", path));
        let mut tok: BpeTokenizer = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("Invalid tokenizer JSON ({}): {}", path, e));
        tok.rebuild_indexes();
        println!("Loaded tokenizer from {} — vocab size: {}", path, tok.vocab.len());
        tok
    }

    /// Encode a full corpus into training documents: each line -> [BOS, ..tokens.., EOS]
    fn encode_corpus(&self, text: &str) -> Vec<Vec<usize>> {
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(|line| {
                let mut toks = vec![self.bos_id];
                toks.extend(self.encode(line));
                toks.push(self.eos_id);
                toks
            })
            .collect()
    }
}

// ============================================================================
// DATA PREPARATION — uses BPE tokenizer
// ============================================================================

struct TrainingData {
    docs: Vec<Vec<usize>>,  // tokenized documents (each is [BOS, ..tokens.., EOS])
    vocab_size: usize,
    bos: usize,
    eos: usize,
    tokenizer: BpeTokenizer,
}

fn prepare_data(text: &str, tokenizer: BpeTokenizer) -> TrainingData {
    let docs = tokenizer.encode_corpus(text);
    let vocab_size = tokenizer.vocab_size();
    let bos = tokenizer.bos_id;
    let eos = tokenizer.eos_id;

    let total_tokens: usize = docs.iter().map(|d| d.len()).sum();
    println!("num docs: {}", docs.len());
    println!("total tokens: {}", total_tokens);
    println!("vocab size: {}", vocab_size);

    TrainingData { docs, vocab_size, bos, eos, tokenizer }
}

// ============================================================================
// TRAIN TOKENIZER — standalone command
// ============================================================================

fn run_train_tokenizer(config: &Config) {
    println!("\n{}", "=".repeat(70));
    println!("   BPE TOKENIZER TRAINING");
    println!("{}\n", "=".repeat(70));

    let text = if std::path::Path::new(&config.training_file).exists() {
        fs::read_to_string(&config.training_file).expect("Failed to read training file")
    } else {
        eprintln!("Training file '{}' not found.", config.training_file);
        eprintln!("Usage: gpt-rust train-tokenizer --trainingFile=input.txt --vocabSize=512");
        std::process::exit(1);
    };

    let tokenizer = BpeTokenizer::train(&text, config.target_vocab_size);
    tokenizer.save(&config.tokenizer_path);

    // Print a few encoding examples
    println!("\n--- encoding examples ---");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(5).collect();
    for line in &lines {
        let ids = tokenizer.encode(line.trim());
        let decoded = tokenizer.decode(&ids);
        println!("  \"{}\" -> {} tokens -> \"{}\"", line.trim(), ids.len(), decoded);
    }
    println!();
}

// ============================================================================
// TRAINING
// ============================================================================

fn run_train(config: &Config) {
    println!("\n{}", "=".repeat(70));
    println!("   GPT + GQA + MoE — VECTORIZED AUTODIFF + BLAS + RAYON (2026)");
    println!("{}\n", "=".repeat(70));

    // Load data
    let text = if std::path::Path::new(&config.training_file).exists() {
        fs::read_to_string(&config.training_file).expect("Failed to read training file")
    } else {
        // Try to download names dataset
        eprintln!("Training file '{}' not found.", config.training_file);
        eprintln!("Please provide a training file: --trainingFile=input.txt");
        eprintln!("For the classic GPT-names experiment, download:");
        eprintln!("  https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");
        std::process::exit(1);
    };

    // Load tokenizer (must be trained first via train-tokenizer command)
    if !std::path::Path::new(&config.tokenizer_path).exists() {
        eprintln!("Error: tokenizer file '{}' not found.", config.tokenizer_path);
        eprintln!("Train a tokenizer first:");
        eprintln!("  gpt-rust train-tokenizer --trainingFile=input.txt --vocabSize=512");
        std::process::exit(1);
    }
    let tokenizer = BpeTokenizer::load(&config.tokenizer_path);
    let data = prepare_data(&text, tokenizer);

    println!("Config: n_embd={} n_head={} n_kv_head={} (GQA group={}) n_layer={} block_size={}",
        config.n_embd, config.n_head, config.n_kv_head,
        config.n_head / config.n_kv_head, config.n_layer, config.block_size);
    println!("  MoE: n_experts={} (1 shared + {} sparse) top_k={}  lr={} steps={}",
        config.n_experts, config.n_experts - 1, config.top_k,
        config.learning_rate, config.num_steps);
    println!("  Tokenizer: {} (vocab={})", config.tokenizer_path, data.vocab_size);

    let mut rng = rand::thread_rng();
    let mut g = Graph::new();

    let gpt = if !config.load_checkpoint.is_empty() && std::path::Path::new(&config.load_checkpoint).exists() {
        let bytes = fs::read(&config.load_checkpoint).unwrap();
        let cp: Checkpoint = bincode::deserialize(&bytes).unwrap();
        let c = &cp.config;
        GPT::new(c.vocab_size, c.n_embd, c.n_head, c.n_kv_head, c.n_layer, c.block_size,
            c.n_experts, c.top_k, &mut g, &mut rng)
    } else {
        GPT::new(data.vocab_size, config.n_embd, config.n_head, config.n_kv_head, config.n_layer,
            config.block_size, config.n_experts, config.top_k, &mut g, &mut rng)
    };

    g.freeze_params();
    let pids = gpt.param_ids();
    let mut ps = ParamSet::new(pids.clone(), &g);
    let mut opt = Adam::new(config.learning_rate, config.beta1, config.beta2, config.weight_decay);

    if !config.load_checkpoint.is_empty() && std::path::Path::new(&config.load_checkpoint).exists() {
        let cp = gpt.load_checkpoint(&config.load_checkpoint, &mut g, &mut ps);
        if cp.optimizer_step > 0 { opt.t = cp.optimizer_step; }
    }

    println!("num params: {}", gpt.total_params(&g));
    println!("Engine: Vectorized Tensor Autodiff + matrixmultiply BLAS + Rayon parallel GQA + MoE");
    println!("\nTraining...\n");

    let t0 = Instant::now();
    let num_docs = data.docs.len();

    // Shuffle order
    let mut doc_order: Vec<usize> = (0..num_docs).collect();

    for step in 0..config.num_steps {
        // Shuffle docs periodically
        if step % num_docs == 0 {
            for i in (1..doc_order.len()).rev() {
                let j = rng.gen_range(0..=i);
                doc_order.swap(i, j);
            }
        }

        let doc_idx = doc_order[step % num_docs];
        let tokens = &data.docs[doc_idx];
        let n = config.block_size.min(tokens.len() - 1);
        if n == 0 { continue; }

        let input_tokens = &tokens[..n];
        let target_tokens: Vec<usize> = tokens[1..n + 1].to_vec();

        // Forward
        g.reset();
        g.zero_grad();
        let logits = gpt.forward(input_tokens, &mut g);
        let loss = gpt.cross_entropy_loss(&logits, &target_tokens, &mut g);

        // Backward
        g.backward(loss);
        GPT::clip_grad(&mut g, &pids, 1.0);

        // Linear LR decay (matching Python)
        let lr_t = config.learning_rate * (1.0 - step as f32 / config.num_steps as f32);
        opt.step(&mut g, &mut ps, Some(lr_t));

        let loss_val = g.data(loss)[0];
        if (step + 1) % config.log_every == 0 || step == 0 {
            println!("step {:4} / {:4} | loss {:.4} | {:.1}s",
                step + 1, config.num_steps, loss_val, t0.elapsed().as_secs_f64());
        }

        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let cp = config.save_path.replace(".bin", &format!("_step{}.bin", step + 1));
            gpt.save_checkpoint(&cp, serde_json::json!({"step": step+1, "loss": loss_val}), opt.t, &g, &ps);
        }
    }

    let tt = t0.elapsed().as_secs_f64();
    println!("\nTraining done in {:.1}s\n", tt);

    if config.save_on_complete {
        gpt.save_checkpoint(&config.save_path,
            serde_json::json!({"steps": config.num_steps, "time": format!("{:.1}", tt), "completed": true}),
            opt.t, &g, &ps);
    }

    // Generate samples
    println!("--- inference (new, hallucinated names) ---");
    generate_samples(&gpt, &data, config, &mut g, &mut rng);
}

// ============================================================================
// GENERATION — autoregressive sampling with temperature
// ============================================================================

fn generate_samples(gpt: &GPT, data: &TrainingData, config: &Config, g: &mut Graph, rng: &mut impl Rng) {
    let temperature = config.temperature;
    let gen_length = if config.gen_length > 0 { config.gen_length } else { gpt.block_size };

    for sample_idx in 0..config.num_samples {
        let mut tokens = vec![data.bos]; // Start with BOS

        for _pos in 0..gen_length {
            g.reset();
            // Feed all tokens so far through the model
            let seq = if tokens.len() > gpt.block_size {
                &tokens[tokens.len() - gpt.block_size..]
            } else {
                &tokens
            };

            let logits_all = gpt.forward(seq, g);
            // Take logits for the last position
            let last_logits = logits_all.last().unwrap();
            let ld = g.data(*last_logits).to_vec();

            // Temperature-scaled softmax sampling
            let scaled: Vec<f32> = ld.iter().map(|l| l / temperature).collect();
            let mx = vec_max(&scaled);
            let exps: Vec<f32> = scaled.iter().map(|l| (l - mx).exp()).collect();
            let sm: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sm).collect();

            // Weighted random sampling
            let mut rv: f32 = rng.gen();
            let mut next_tok = 0;
            for j in 0..probs.len() {
                rv -= probs[j];
                if rv <= 0.0 { next_tok = j; break; }
            }

            // BOS or EOS = end of sequence
            if next_tok == data.bos || next_tok == data.eos { break; }
            tokens.push(next_tok);
        }

        // Decode generated tokens (skip the leading BOS)
        let generated = data.tokenizer.decode(&tokens[1..]);
        println!("sample {:2}: {}", sample_idx + 1, generated);
    }
}

// ============================================================================
// INFERENCE — load checkpoint and generate
// ============================================================================

fn run_inference(config: &Config) {
    println!("\n{}", "=".repeat(70));
    println!("   GPT INFERENCE MODE");
    println!("{}\n", "=".repeat(70));

    if config.load_checkpoint.is_empty() {
        eprintln!("Error: --loadCheckpoint is required for inference.");
        eprintln!("Usage: gpt-rust inference --loadCheckpoint=./gpt_checkpoint.bin --tokenizerPath=./tokenizer.json");
        std::process::exit(1);
    }

    if !std::path::Path::new(&config.tokenizer_path).exists() {
        eprintln!("Error: tokenizer file '{}' not found.", config.tokenizer_path);
        eprintln!("Provide the tokenizer used during training: --tokenizerPath=./tokenizer.json");
        std::process::exit(1);
    }
    let tokenizer = BpeTokenizer::load(&config.tokenizer_path);
    let text = String::new(); // inference doesn't need training text
    let data = prepare_data(&text, tokenizer);

    let mut rng = rand::thread_rng();
    let mut g = Graph::new();

    let bytes = fs::read(&config.load_checkpoint).unwrap();
    let cp: Checkpoint = bincode::deserialize(&bytes).unwrap();
    let c = &cp.config;
    let gpt = GPT::new(c.vocab_size, c.n_embd, c.n_head, c.n_kv_head, c.n_layer, c.block_size,
        c.n_experts, c.top_k, &mut g, &mut rng);

    g.freeze_params();
    let mut ps = ParamSet::new(gpt.param_ids(), &g);
    gpt.load_checkpoint(&config.load_checkpoint, &mut g, &mut ps);

    println!("Temperature: {}\n", config.temperature);
    println!("--- inference (hallucinated names) ---");
    generate_samples(&gpt, &data, config, &mut g, &mut rng);
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let config = Config::from_args();

    match config.command.as_str() {
        "train-tokenizer" | "train_tokenizer" | "bpe" => run_train_tokenizer(&config),
        "train" => run_train(&config),
        "inference" | "generate" | "infer" => run_inference(&config),
        _ => {
            println!("\n{}", "=".repeat(70));
            println!("   GPT + GQA + MoE + RoPE + BPE — Rust (2026)");
            println!("   Vectorized Tensor Autodiff + matrixmultiply BLAS + Rayon");
            println!("{}", "=".repeat(70));
            println!("\nUsage: gpt-rust <command> [options]\n");
            println!("Commands:");
            println!("  train-tokenizer   Train a BPE tokenizer on a text file");
            println!("  train             Train the GPT model (requires a trained tokenizer)");
            println!("  inference         Generate text from a trained checkpoint\n");
            println!("Tokenizer Examples:");
            println!("  gpt-rust train-tokenizer --trainingFile=input.txt --vocabSize=512");
            println!("  gpt-rust train-tokenizer --trainingFile=input.txt --vocabSize=1024 --tokenizerPath=./my_tok.json\n");
            println!("Training Examples:");
            println!("  gpt-rust train --trainingFile=input.txt --numSteps=1000");
            println!("  gpt-rust train --trainingFile=input.txt --n_embd=64 --n_head=4 --n_layer=2");
            println!("  gpt-rust train --loadCheckpoint=./gpt_checkpoint.bin --numSteps=2000  (resume)\n");
            println!("Inference Examples:");
            println!("  gpt-rust inference --loadCheckpoint=./gpt_checkpoint.bin --tokenizerPath=./tokenizer.json");
            println!("  gpt-rust inference --loadCheckpoint=./gpt_checkpoint.bin --tokenizerPath=./tokenizer.json --temperature=0.8 --numSamples=30\n");
            println!("All Options:");
            println!("  Model:      --n_embd=16 --n_head=4 --n_kv_head=2 --n_layer=1 --blockSize=16");
            println!("  MoE:        --n_experts=4 --top_k=2  (expert 0 = always-on shared)");
            println!("  Tokenizer:  --tokenizerPath=./tokenizer.json --vocabSize=512");
            println!("  Training:   --numSteps=1000 --lr=0.01 --beta1=0.85 --beta2=0.99 --wd=0.0");
            println!("  Data:       --trainingFile=input.txt");
            println!("  Checkpoint: --savePath=./gpt_checkpoint.bin --saveEvery=100 --loadCheckpoint=./model.bin");
            println!("  Inference:  --temperature=0.5 --numSamples=20 --genLength=16");
            println!("  Output:     --logEvery=1");

            // If no command but training file exists, default to train-tokenizer + train
            if std::path::Path::new(&config.training_file).exists() {
                println!("\n[Defaulting to train mode since {} exists]", config.training_file);
                if !std::path::Path::new(&config.tokenizer_path).exists() {
                    println!("[No tokenizer found — training one first with vocab={}]", config.target_vocab_size);
                    run_train_tokenizer(&config);
                }
                run_train(&config);
            }
        }
    }
}
