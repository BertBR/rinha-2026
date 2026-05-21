#ifndef RINHA_KNN_H
#define RINHA_KNN_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the embedded IVF index and run warmup queries.
// Must be called exactly once before any rinha_knn_fraud_count call.
void rinha_knn_init(void);

// Query the kNN index with 14 f32 values. Returns the count of fraud-labeled
// neighbors (0..5) in the top-5 nearest references.
unsigned char rinha_knn_fraud_count(const float* q);

#ifdef __cplusplus
}
#endif

#endif
