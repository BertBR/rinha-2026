package knn

/*
#cgo CFLAGS: -I${SRCDIR}/../../core
#cgo LDFLAGS: -L${SRCDIR}/../../core/target/release -lrinha_knn -Wl,-rpath,${SRCDIR}/../../core/target/release -Wl,--unresolved-symbols=ignore-in-shared-libs
#include "rinha_knn.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

var initOnce sync.Once

func Init() {
	initOnce.Do(func() {
		C.rinha_knn_init()
	})
}

// FraudCount returns the count of fraud-labeled neighbors (0..5) for the
// given 14-dimensional query vector. Init must have been called first.
func FraudCount(q *[14]float32) uint8 {
	return uint8(C.rinha_knn_fraud_count((*C.float)(unsafe.Pointer(&q[0]))))
}
