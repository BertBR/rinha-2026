// Loads the IVF index file at process start.
//
// File format (little-endian):
//   magic         4 bytes  "RKNN"
//   version       4 bytes  u32 = 2  (v1 omits bucket_radius)
//   n             4 bytes  u32  number of vectors
//   k             4 bytes  u32  number of centroids
//   d             4 bytes  u32  dimensions (always 14)
//   _reserved     12 bytes
//   centroids     k * 14 * 4 bytes   f32 centroid coordinates (row-major)
//   bucket_off    (k + 1) * 4 bytes  u32 cumulative offsets into bucket_vec
//   bucket_radius k * 4 bytes        f32 max dist (in i16 quantized space)
//                                    from centroid to any vec in bucket
//   bucket_label  n bytes            u8 labels (1 = fraud, 0 = legit), in bucket order
//   bucket_orig   n * 4 bytes        u32 original index (for grader tie-break)
//   bucket_vec    n * 14 * 2 bytes   i16 quantized vectors in bucket order
//
// All numeric arrays are not gzipped at the file level (the gzipping is done
// at the .gz wrapping level if present). The runtime path uses
// `include_bytes!` of the gzipped file and inflates in-memory at startup.

use flate2::read::GzDecoder;
use std::io::Read;
use std::sync::OnceLock;

pub const DIMS: usize = 14;
pub const QUANT_SCALE: f32 = 10000.0;

#[repr(C)]
pub struct Dataset {
    pub n: usize,
    pub k: usize,
    pub centroids: Vec<f32>,       // raw f32 (kept for diagnostics)
    pub centroids_i16: Vec<i16>,   // same centroids quantized to i16 for pruning
    pub bucket_off: Vec<u32>,
    pub bucket_radius: Vec<f32>,   // v2: max sqrt(i16-dist²) per bucket
    pub max_radius: f32,           // max(bucket_radius) for global pruning
    pub bucket_label: Vec<u8>,
    pub bucket_orig: Vec<u32>,
    pub bucket_vec: Vec<i16>,
}

static DATASET: OnceLock<Dataset> = OnceLock::new();

pub fn dataset() -> &'static Dataset {
    DATASET.get().expect("dataset not initialized")
}

pub fn init() {
    let ds = load_embedded().expect("failed to load IVF index");
    if DATASET.set(ds).is_err() {
        panic!("dataset already initialized");
    }
}

fn load_embedded() -> std::io::Result<Dataset> {
    static INDEX_GZ: &[u8] = include_bytes!("../data/index.bin.gz");
    let mut gz = GzDecoder::new(INDEX_GZ);

    let mut hdr = [0u8; 32];
    gz.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RKNN" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    if version != 1 && version != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported version",
        ));
    }
    let n = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    if d != DIMS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "wrong dims",
        ));
    }

    let mut centroids = vec![0f32; k * DIMS];
    read_into_f32(&mut gz, &mut centroids)?;

    let mut bucket_off = vec![0u32; k + 1];
    read_into_u32(&mut gz, &mut bucket_off)?;

    let total = bucket_off[k] as usize;
    if total != n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bucket totals mismatch",
        ));
    }

    // bucket_radius is only present in v2. For v1, leave zeros — pruning
    // becomes a no-op then (lb = sqrt(cdist) - 0 = sqrt(cdist), which is
    // never > heap.worst when the closest centroid is closest, so all
    // buckets get scanned — equivalent to old behaviour).
    let mut bucket_radius = vec![0f32; k];
    if version >= 2 {
        read_into_f32(&mut gz, &mut bucket_radius)?;
    }

    let mut bucket_label = vec![0u8; n];
    gz.read_exact(&mut bucket_label)?;

    let mut bucket_orig = vec![0u32; n];
    read_into_u32(&mut gz, &mut bucket_orig)?;

    // Read 14-dim vectors from the index. Allocate 8 extra zero i16 at the
    // tail so the AVX2 scan kernel can _mm_loadu_si128(vp + vbase + 8) on
    // the LAST vec without going OOB — that tail load reads 8 i16 starting
    // at (n-1)*14 + 8, which goes 2 i16 past the real data. The pad makes
    // it safe; the kernel masks those lanes off via tail_mask.
    let mut bucket_vec = vec![0i16; n * DIMS + 8];
    {
        let body = &mut bucket_vec[..n * DIMS];
        read_into_i16(&mut gz, body)?;
    }

    // centroids_i16 was used by the bbox-prune kNN variant; the NPROBE
    // escalate flow (r6/r16) uses the f32 centroid_distances kernel, so
    // we keep this allocation tiny — just a length-zero stub to satisfy
    // the Dataset struct shape. Saves ~64KB at runtime.
    let centroids_i16: Vec<i16> = Vec::new();

    let max_radius = bucket_radius.iter().cloned().fold(0f32, f32::max);

    Ok(Dataset {
        n,
        k,
        bucket_radius,
        max_radius,
        centroids,
        centroids_i16,
        bucket_off,
        bucket_label,
        bucket_orig,
        bucket_vec,
    })
}

fn read_into_f32<R: Read>(r: &mut R, dst: &mut [f32]) -> std::io::Result<()> {
    let byte_len = std::mem::size_of_val(dst);
    let raw = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, byte_len) };
    r.read_exact(raw)
}

fn read_into_u32<R: Read>(r: &mut R, dst: &mut [u32]) -> std::io::Result<()> {
    let byte_len = std::mem::size_of_val(dst);
    let raw = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, byte_len) };
    r.read_exact(raw)
}

fn read_into_i16<R: Read>(r: &mut R, dst: &mut [i16]) -> std::io::Result<()> {
    let byte_len = std::mem::size_of_val(dst);
    let raw = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, byte_len) };
    r.read_exact(raw)
}
