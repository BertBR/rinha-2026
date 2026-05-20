// rinha-knn: native kNN core for Rinha de Backend 2026.
//
// Architecture:
//   - IVF index with ~2048 centroids (built at Docker build time, see
//     build_ivf.rs).
//   - Each vector quantized to 14 × i16 with scale 10000 (matches source
//     data precision of 4 decimal places).
//   - At query time, scan top-NPROBE centroid buckets with AVX2 SIMD.
//   - Two-stage: FAST_NPROBE=8 covers most queries; if the result lands on
//     the 2/5 or 3/5 fraud-count boundary (sensitive to misranked neighbors),
//     escalate to FULL_NPROBE=32 for confirmation.
//
// Exposes two N-API functions to Node:
//   initKnn()                                    -> ()
//   knnFraudCount(v0, v1, ..., v13: f64)         -> u8 (count of frauds in top-5)

mod data;
mod knn;

use napi_derive::napi;

#[napi]
pub fn init_knn() {
    data::init();
    knn::warmup();
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn knn_fraud_count(
    v0: f64,
    v1: f64,
    v2: f64,
    v3: f64,
    v4: f64,
    v5: f64,
    v6: f64,
    v7: f64,
    v8: f64,
    v9: f64,
    v10: f64,
    v11: f64,
    v12: f64,
    v13: f64,
) -> u8 {
    let q: [f32; 14] = [
        v0 as f32, v1 as f32, v2 as f32, v3 as f32, v4 as f32, v5 as f32, v6 as f32, v7 as f32,
        v8 as f32, v9 as f32, v10 as f32, v11 as f32, v12 as f32, v13 as f32,
    ];
    knn::query(&q, data::dataset())
}
