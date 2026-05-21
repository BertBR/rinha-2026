// Fraud-detection API for Rinha de Backend 2026.
//
// HTTP server bound to a Unix domain socket. On POST /fraud-score, it
// parses the fixed-schema payload into a 14-dimensional float32 vector,
// runs an IVF kNN query through the cgo-bound Rust core, and writes one
// of six precomputed JSON responses indexed by fraud count (0..5).

package main

import (
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/BertBR/rinha-2026/internal/knn"
	"github.com/BertBR/rinha-2026/internal/vec"
)

// Six prebaked JSON bodies, indexed by fraud count (count/5 = fraud_score).
var bodyBuffers = [][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1.0}`),
}

type reqState struct {
	q   [14]float32
	ctx *vec.ParseContext
}

var statePool sync.Pool

var (
	normalization *vec.Normalization
	mccRisk       vec.MccRiskMap

	// ready flips to true once the native kNN index is loaded and warm.
	// readyCh is closed at the same moment so the first /fraud-score
	// request can wait synchronously without busy-spinning.
	ready   atomic.Bool
	readyCh = make(chan struct{})
)

func main() {
	dataDir := envOr("DATA_DIR", "./data")
	udsPath := envOr("UDS_PATH", "/sockets/api.sock")

	if err := loadConfig(dataDir); err != nil {
		log.Fatalf("[api] load config: %v", err)
	}
	statePool.New = func() any {
		return &reqState{ctx: vec.NewContext(normalization, mccRisk)}
	}

	// Open the UDS listener BEFORE the heavy native init so HAProxy's TCP
	// healthcheck connects immediately. The Rust init (gunzip 42 MB embedded
	// index + 500-query warmup) takes 5-15 s under the 0.42 CPU cgroup, and
	// the rinha smoke test marks the stack DOWN if the LB has no upstream
	// when k6 fires. Handler blocks on `readyCh` until init completes.
	if err := os.RemoveAll(udsPath); err != nil && !os.IsNotExist(err) {
		log.Fatalf("[api] cleanup uds: %v", err)
	}
	ln, err := net.Listen("unix", udsPath)
	if err != nil {
		log.Fatalf("[api] listen %s: %v", udsPath, err)
	}
	if err := os.Chmod(udsPath, 0o666); err != nil {
		log.Printf("[api] chmod uds: %v", err)
	}

	go func() {
		log.Printf("[api] initializing native kNN")
		t0 := time.Now()
		knn.Init()
		log.Printf("[api]   ready in %v", time.Since(t0).Round(time.Millisecond))
		ready.Store(true)
		close(readyCh)
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", handleReady)
	mux.HandleFunc("POST /fraud-score", handleFraudScore)

	srv := &http.Server{
		Handler:      mux,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 30 * time.Second, // generous so the first request can wait through init
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("[api] listening on UDS %s (GOMAXPROCS=%d)", udsPath, runtime.GOMAXPROCS(0))
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[api] serve: %v", err)
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func loadConfig(dir string) error {
	normRaw, err := os.ReadFile(dir + "/normalization.json")
	if err != nil {
		return err
	}
	mccRaw, err := os.ReadFile(dir + "/mcc_risk.json")
	if err != nil {
		return err
	}
	normalization = &vec.Normalization{}
	if err := json.Unmarshal(normRaw, normalization); err != nil {
		return err
	}
	mccRisk = vec.MccRiskMap{}
	if err := json.Unmarshal(mccRaw, &mccRisk); err != nil {
		return err
	}
	return nil
}

func handleReady(w http.ResponseWriter, _ *http.Request) {
	if ready.Load() {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	// First request(s) arriving during native init wait for it to finish.
	// k6's per-request timeout is 10 s, init is typically 5-15 s on the
	// throttled container, so the first call may be slow but every subsequent
	// one is fast.
	if !ready.Load() {
		select {
		case <-readyCh:
		case <-r.Context().Done():
			return
		}
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(bodyBuffers[0])
		return
	}

	st := statePool.Get().(*reqState)
	defer statePool.Put(st)

	var count uint8
	if vec.ParseAndVectorize(body, &st.q, st.ctx) {
		count = knn.FraudCount(&st.q)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(bodyBuffers[count])
}
