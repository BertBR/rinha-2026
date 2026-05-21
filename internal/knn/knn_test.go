package knn

import "testing"

// Smoke test: init the native side and query with an all-zero vector.
// Just verifies the cgo path works end to end; the exact fraud count
// for an all-zero query is whatever the IVF index returns.
func TestSmoke(t *testing.T) {
	Init()
	var q [14]float32
	got := FraudCount(&q)
	t.Logf("FraudCount(zero vector) = %d", got)
	if got > 5 {
		t.Fatalf("fraud count out of range: %d", got)
	}
}

// Determinism: same query yields same count.
func TestDeterminism(t *testing.T) {
	Init()
	q := [14]float32{0.5, 0.3, 0.1, 0.7, 0.5, -1, -1, 0.2, 0.6, 1, 0, 0, 0.4, 0.5}
	a := FraudCount(&q)
	b := FraudCount(&q)
	if a != b {
		t.Fatalf("non-deterministic: %d vs %d", a, b)
	}
}

func BenchmarkFraudCount(b *testing.B) {
	Init()
	q := [14]float32{0.5, 0.3, 0.1, 0.7, 0.5, -1, -1, 0.2, 0.6, 1, 0, 0, 0.4, 0.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FraudCount(&q)
	}
}
