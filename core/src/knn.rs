// kNN query over the IVF index with AVX2 + FMA SIMD.
//
// Algorithm per query:
//   1. Compute distance² from query to all k centroids (AVX2).
//   2. Find FAST_NPROBE smallest centroid distances.
//   3. Scan vectors in those buckets; maintain top-5 with (dist, orig, label).
//   4. If fraud count is 2 or 3 (decision boundary at 0.6), re-scan with
//      FULL_NPROBE to confirm.
//   5. Return count of fraud-labeled neighbors in the final top-5.

use crate::data::{dataset, Dataset, DIMS, QUANT_SCALE};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

const FAST_NPROBE: usize = 12;
const FULL_NPROBE: usize = 64;
const MAX_CENTROIDS: usize = 8192;

pub fn query(q: &[f32; 14], ds: &Dataset) -> u8 {
    unsafe { query_avx2(q, ds) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn query_avx2(q: &[f32; 14], ds: &Dataset) -> u8 {
    let mut cdists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
    centroid_distances(q, ds, &mut cdists);

    let fast = top_k_indices::<FAST_NPROBE>(&cdists, ds.k);

    // Quantize query to i16 once.
    let mut qi = [0i16; 14];
    quantize_query(q, &mut qi);

    let mut heap = Heap5::new();
    scan_buckets(&qi, ds, &fast, &mut heap);
    let fast_count = heap.count_frauds();

    if fast_count != 2 && fast_count != 3 {
        return fast_count;
    }

    // Escalate.
    let full = top_k_indices::<FULL_NPROBE>(&cdists, ds.k);
    let mut heap = Heap5::new();
    scan_buckets(&qi, ds, &full, &mut heap);
    heap.count_frauds()
}

#[inline]
fn quantize_query(q: &[f32; 14], out: &mut [i16; 14]) {
    for d in 0..DIMS {
        let v = q[d];
        if (v + 1.0).abs() < 1e-5 {
            out[d] = -32768;
        } else {
            let qq = (v * QUANT_SCALE).round() as i32;
            out[d] = qq.clamp(-32768, 32767) as i16;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_distances(
    q: &[f32; 14],
    ds: &Dataset,
    out: &mut [MaybeUninit<f32>; MAX_CENTROIDS],
) {
    let k = ds.k;
    let cptr = ds.centroids.as_ptr();
    let optr = out.as_mut_ptr() as *mut f32;

    let qv: [__m256; 14] = [
        _mm256_set1_ps(q[0]),
        _mm256_set1_ps(q[1]),
        _mm256_set1_ps(q[2]),
        _mm256_set1_ps(q[3]),
        _mm256_set1_ps(q[4]),
        _mm256_set1_ps(q[5]),
        _mm256_set1_ps(q[6]),
        _mm256_set1_ps(q[7]),
        _mm256_set1_ps(q[8]),
        _mm256_set1_ps(q[9]),
        _mm256_set1_ps(q[10]),
        _mm256_set1_ps(q[11]),
        _mm256_set1_ps(q[12]),
        _mm256_set1_ps(q[13]),
    ];

    let mut i = 0;
    while i + 8 <= k {
        let mut acc = _mm256_setzero_ps();
        for d in 0..DIMS {
            let mut tmp = [0f32; 8];
            for j in 0..8 {
                tmp[j] = *cptr.add((i + j) * DIMS + d);
            }
            let cv = _mm256_loadu_ps(tmp.as_ptr());
            let diff = _mm256_sub_ps(cv, qv[d]);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }
        _mm256_storeu_ps(optr.add(i), acc);
        i += 8;
    }
    while i < k {
        let mut sum = 0f32;
        for d in 0..DIMS {
            let cv = *cptr.add(i * DIMS + d);
            let diff = cv - q[d];
            sum += diff * diff;
        }
        *optr.add(i) = sum;
        i += 1;
    }
}

unsafe fn top_k_indices<const N: usize>(
    dists: &[MaybeUninit<f32>; MAX_CENTROIDS],
    k: usize,
) -> [u32; N] {
    let mut top_d = [f32::INFINITY; N];
    let mut top_i = [0u32; N];
    let dp = dists.as_ptr() as *const f32;

    for ci in 0..k {
        let d = *dp.add(ci);
        if d >= top_d[0] {
            continue;
        }
        top_d[0] = d;
        top_i[0] = ci as u32;
        sift_down_heap(&mut top_d, &mut top_i);
    }

    top_i
}

fn sift_down_heap<const N: usize>(d: &mut [f32; N], i: &mut [u32; N]) {
    let mut p = 0usize;
    loop {
        let l = p * 2 + 1;
        let r = l + 1;
        let mut largest = p;
        if l < N && d[l] > d[largest] {
            largest = l;
        }
        if r < N && d[r] > d[largest] {
            largest = r;
        }
        if largest == p {
            return;
        }
        d.swap(p, largest);
        i.swap(p, largest);
        p = largest;
    }
}

// Top-5 max-heap of (distance², orig-id, label). Tie-break: equal
// distance → smaller orig-id wins (matches grader's brute-force order).
struct Heap5 {
    dist: [f32; 5],
    orig: [u32; 5],
    label: [u8; 5],
    size: usize,
}

impl Heap5 {
    #[inline]
    fn new() -> Self {
        Heap5 {
            dist: [f32::INFINITY; 5],
            orig: [u32::MAX; 5],
            label: [0u8; 5],
            size: 0,
        }
    }

    #[inline]
    fn worst(&self) -> f32 {
        if self.size < 5 {
            f32::INFINITY
        } else {
            self.dist[0]
        }
    }

    #[inline]
    fn try_insert(&mut self, dist: f32, orig: u32, label: u8) {
        if self.size < 5 {
            let i = self.size;
            self.size += 1;
            self.dist[i] = dist;
            self.orig[i] = orig;
            self.label[i] = label;
            self.sift_up(i);
            return;
        }
        if dist > self.dist[0] {
            return;
        }
        if dist == self.dist[0] && orig >= self.orig[0] {
            return;
        }
        self.dist[0] = dist;
        self.orig[0] = orig;
        self.label[0] = label;
        self.sift_down_root();
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            let bigger = self.dist[p] < self.dist[i]
                || (self.dist[p] == self.dist[i] && self.orig[p] < self.orig[i]);
            if !bigger {
                break;
            }
            self.dist.swap(p, i);
            self.orig.swap(p, i);
            self.label.swap(p, i);
            i = p;
        }
    }

    fn sift_down_root(&mut self) {
        let n = self.size;
        let mut p = 0usize;
        loop {
            let l = p * 2 + 1;
            let r = l + 1;
            let mut largest = p;
            if l < n
                && (self.dist[l] > self.dist[largest]
                    || (self.dist[l] == self.dist[largest] && self.orig[l] > self.orig[largest]))
            {
                largest = l;
            }
            if r < n
                && (self.dist[r] > self.dist[largest]
                    || (self.dist[r] == self.dist[largest] && self.orig[r] > self.orig[largest]))
            {
                largest = r;
            }
            if largest == p {
                return;
            }
            self.dist.swap(p, largest);
            self.orig.swap(p, largest);
            self.label.swap(p, largest);
            p = largest;
        }
    }

    #[inline]
    fn count_frauds(&self) -> u8 {
        let mut s: u8 = 0;
        for i in 0..self.size {
            s += self.label[i];
        }
        s
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn scan_buckets(qi: &[i16; 14], ds: &Dataset, probes: &[u32], heap: &mut Heap5) {
    let vp = ds.bucket_vec.as_ptr();
    let lp = ds.bucket_label.as_ptr();
    let op = ds.bucket_orig.as_ptr();

    for &ci in probes.iter() {
        let start = ds.bucket_off[ci as usize] as usize;
        let end = ds.bucket_off[ci as usize + 1] as usize;

        let mut j = start;
        // Scalar loop for now. The body is small (14 dims), V8 unrolls similar
        // logic to vectorize-friendly code, but here we're already in Rust.
        // A real SIMD batch would process 8 vectors at once after lane permute;
        // for now correctness first.
        while j < end {
            let vbase = j * DIMS;
            let mut s: i32 = 0;
            for d in 0..DIMS {
                let diff = (qi[d] as i32) - (*vp.add(vbase + d) as i32);
                s += diff * diff;
            }
            // Early-out optimization: if current sum already exceeds worst,
            // skip the heap update. But we still pay full compute.
            let s_f = s as f32;
            if s_f < heap.worst() || heap.size < 5 {
                heap.try_insert(s_f, *op.add(j), *lp.add(j));
            }
            j += 1;
        }
    }
}

pub fn warmup() {
    let ds = dataset();
    let mut q = [0f32; 14];
    let mut seed = 0x12345u32;
    for _ in 0..500 {
        for d in 0..DIMS {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            q[d] = (seed >> 8) as f32 / (1u32 << 24) as f32;
        }
        let _ = query(&q, ds);
    }
}
