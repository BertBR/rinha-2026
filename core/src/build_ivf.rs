// Build the IVF index from references.json.gz at Docker build time.
//
// Workflow:
//   1. Stream-parse references.json.gz into (vector: [f32; 14], label, orig_id).
//   2. Run mini-batch k-means clustering with K centroids (default 2048),
//      M iterations (default 3) — produces "good enough" centroids.
//   3. Assign each vector to its nearest centroid.
//   4. Lay out vectors by bucket (cluster) order; emit RKNN index file
//      (header + centroids + bucket offsets + labels + orig ids + i16 vectors).
//   5. gzip the file.
//
// Usage:
//   build-ivf <references.json.gz> <output index.bin.gz> [K] [M]

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

const DIMS: usize = 14;
const DEFAULT_K: usize = 2048;
const DEFAULT_ITERS: usize = 3;
const QUANT_SCALE: f32 = 10000.0;

#[derive(Deserialize)]
struct Record {
    vector: [f32; 14],
    label: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: build-ivf <input.json.gz> <output.bin.gz> [K] [iters]");
        std::process::exit(2);
    }
    let input = &args[1];
    let output = &args[2];
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_K);
    let iters: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERS);

    let t0 = Instant::now();
    println!("[build-ivf] loading {input}");

    // 1. Load all vectors into flat f32 array and a label vec.
    let mut vectors: Vec<f32> = Vec::with_capacity(3_000_000 * DIMS);
    let mut labels: Vec<u8> = Vec::with_capacity(3_000_000);

    let file = File::open(input).expect("open references");
    let gz = GzDecoder::new(BufReader::new(file));
    let mut full = String::with_capacity(300 * 1024 * 1024);
    BufReader::new(gz)
        .read_to_string(&mut full)
        .expect("decompress");

    // Brute-force JSON streaming: parse top-level array element by element.
    // We use serde_json::Deserializer for streaming.
    let stream: Vec<Record> = serde_json::from_str(&full).expect("parse json");
    drop(full);

    for r in &stream {
        for d in 0..DIMS {
            vectors.push(r.vector[d]);
        }
        labels.push(if r.label == "fraud" { 1 } else { 0 });
    }
    let n = labels.len();
    drop(stream);

    println!(
        "[build-ivf] loaded {} vectors in {:.1}s",
        n,
        t0.elapsed().as_secs_f32()
    );

    // 2. Initialize centroids by sampling vectors. Use evenly-spaced indices.
    let mut centroids: Vec<f32> = Vec::with_capacity(k * DIMS);
    for ci in 0..k {
        let idx = (ci * n) / k;
        for d in 0..DIMS {
            centroids.push(vectors[idx * DIMS + d]);
        }
    }

    // 3. Run k-means iterations.
    let mut assign: Vec<u32> = vec![0u32; n];
    for it in 0..iters {
        let t_it = Instant::now();
        // Assignment step
        for i in 0..n {
            let vbase = i * DIMS;
            let mut best_d = f32::INFINITY;
            let mut best_c = 0u32;
            for ci in 0..k {
                let cbase = ci * DIMS;
                let mut s = 0f32;
                for d in 0..DIMS {
                    let diff = vectors[vbase + d] - centroids[cbase + d];
                    s += diff * diff;
                    if s > best_d {
                        break;
                    }
                }
                if s < best_d {
                    best_d = s;
                    best_c = ci as u32;
                }
            }
            assign[i] = best_c;
        }
        println!(
            "[build-ivf] iter {} assignment: {:.1}s",
            it,
            t_it.elapsed().as_secs_f32()
        );

        // Update step
        let mut sums: Vec<f64> = vec![0f64; k * DIMS];
        let mut counts: Vec<u32> = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let vbase = i * DIMS;
            let cbase = c * DIMS;
            for d in 0..DIMS {
                sums[cbase + d] += vectors[vbase + d] as f64;
            }
        }
        for ci in 0..k {
            let cnt = counts[ci];
            if cnt == 0 {
                // Re-seed empty centroid to a random vector.
                let idx = (ci * 7919) % n;
                for d in 0..DIMS {
                    centroids[ci * DIMS + d] = vectors[idx * DIMS + d];
                }
                continue;
            }
            for d in 0..DIMS {
                centroids[ci * DIMS + d] = (sums[ci * DIMS + d] / cnt as f64) as f32;
            }
        }
    }

    // 4. Reassign once more (final assignment).
    for i in 0..n {
        let vbase = i * DIMS;
        let mut best_d = f32::INFINITY;
        let mut best_c = 0u32;
        for ci in 0..k {
            let cbase = ci * DIMS;
            let mut s = 0f32;
            for d in 0..DIMS {
                let diff = vectors[vbase + d] - centroids[cbase + d];
                s += diff * diff;
                if s > best_d {
                    break;
                }
            }
            if s < best_d {
                best_d = s;
                best_c = ci as u32;
            }
        }
        assign[i] = best_c;
    }

    // 5. Layout vectors by bucket order.
    let mut bucket_off: Vec<u32> = vec![0u32; k + 1];
    for &c in &assign {
        bucket_off[c as usize + 1] += 1;
    }
    for ci in 0..k {
        bucket_off[ci + 1] += bucket_off[ci];
    }

    let mut cursor: Vec<u32> = bucket_off.clone();
    let mut out_label: Vec<u8> = vec![0u8; n];
    let mut out_orig: Vec<u32> = vec![0u32; n];
    let mut out_vec: Vec<i16> = vec![0i16; n * DIMS];

    for i in 0..n {
        let c = assign[i] as usize;
        let pos = cursor[c] as usize;
        cursor[c] += 1;
        out_label[pos] = labels[i];
        out_orig[pos] = i as u32;
        let vbase = i * DIMS;
        let obase = pos * DIMS;
        for d in 0..DIMS {
            let v = vectors[vbase + d];
            let qi = if (v + 1.0).abs() < 1e-5 {
                -32768i16
            } else {
                let qq = (v * QUANT_SCALE).round() as i32;
                qq.clamp(-32768, 32767) as i16
            };
            out_vec[obase + d] = qi;
        }
    }

    // 6. Write index file.
    println!("[build-ivf] writing {output}");
    let file = File::create(output).expect("create output");
    let mut writer = BufWriter::new(GzEncoder::new(file, Compression::default()));

    // Header
    writer.write_all(b"RKNN").unwrap();
    writer.write_all(&1u32.to_le_bytes()).unwrap(); // version
    writer.write_all(&(n as u32).to_le_bytes()).unwrap();
    writer.write_all(&(k as u32).to_le_bytes()).unwrap();
    writer.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
    writer.write_all(&[0u8; 12]).unwrap(); // reserved

    // Centroids
    let centroids_bytes = unsafe {
        std::slice::from_raw_parts(
            centroids.as_ptr() as *const u8,
            centroids.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(centroids_bytes).unwrap();

    // Bucket offsets
    let off_bytes = unsafe {
        std::slice::from_raw_parts(
            bucket_off.as_ptr() as *const u8,
            bucket_off.len() * std::mem::size_of::<u32>(),
        )
    };
    writer.write_all(off_bytes).unwrap();

    // Labels
    writer.write_all(&out_label).unwrap();

    // Orig IDs
    let orig_bytes = unsafe {
        std::slice::from_raw_parts(
            out_orig.as_ptr() as *const u8,
            out_orig.len() * std::mem::size_of::<u32>(),
        )
    };
    writer.write_all(orig_bytes).unwrap();

    // Quantized vectors
    let vec_bytes = unsafe {
        std::slice::from_raw_parts(
            out_vec.as_ptr() as *const u8,
            out_vec.len() * std::mem::size_of::<i16>(),
        )
    };
    writer.write_all(vec_bytes).unwrap();

    writer.flush().unwrap();
    println!("[build-ivf] done in {:.1}s", t0.elapsed().as_secs_f32());
}
