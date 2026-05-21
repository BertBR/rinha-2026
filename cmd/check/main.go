// Diff my local Go API against test-data.json ground truth.
// Reports payloads where my fraud_score disagrees with expected,
// so I can hand-trace the vectorize/kNN path.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sync/atomic"
	"time"
)

type entry struct {
	Request  json.RawMessage `json:"request"`
	Expected float64         `json:"expected_fraud_score"`
}

type testData struct {
	Entries []entry `json:"entries"`
}

type apiResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

var (
	target = flag.String("url", "http://localhost:9999/fraud-score", "target api url")
	conc   = flag.Int("c", 20, "concurrent requests")
	dataF  = flag.String("data", "data/test-data.json", "test-data.json path")
	limit  = flag.Int("limit", 0, "if >0, only check first N entries")
)

func main() {
	flag.Parse()

	log.Printf("loading %s ...", *dataF)
	raw, err := os.ReadFile(*dataF)
	if err != nil {
		log.Fatal(err)
	}
	var td testData
	if err := json.Unmarshal(raw, &td); err != nil {
		log.Fatal(err)
	}
	log.Printf("%d entries", len(td.Entries))

	n := len(td.Entries)
	if *limit > 0 && *limit < n {
		n = *limit
		td.Entries = td.Entries[:n]
	}

	client := &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{
		MaxIdleConns: *conc, MaxIdleConnsPerHost: *conc, IdleConnTimeout: 60 * time.Second,
	}}

	type mismatch struct {
		idx      int
		expected float64
		got      float64
		req      json.RawMessage
	}

	mm := make(chan mismatch, 1024)
	done := make(chan struct{})
	var mismatches []mismatch
	go func() {
		for m := range mm {
			mismatches = append(mismatches, m)
		}
		close(done)
	}()

	var processed int64
	t0 := time.Now()

	work := make(chan int, *conc*2)
	stop := make(chan struct{})
	go func() {
		for i := range td.Entries {
			work <- i
		}
		close(work)
	}()

	wgN := *conc
	doneWorkers := make(chan struct{}, wgN)
	for w := 0; w < wgN; w++ {
		go func() {
			for i := range work {
				e := td.Entries[i]
				resp, err := client.Post(*target, "application/json", bytes.NewReader(e.Request))
				if err != nil {
					log.Printf("entry %d: %v", i, err)
					atomic.AddInt64(&processed, 1)
					continue
				}
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				var ar apiResponse
				if err := json.Unmarshal(body, &ar); err != nil {
					log.Printf("entry %d: parse response: %v body=%s", i, err, body)
					atomic.AddInt64(&processed, 1)
					continue
				}
				if math.Abs(ar.FraudScore-e.Expected) > 0.01 {
					mm <- mismatch{idx: i, expected: e.Expected, got: ar.FraudScore, req: e.Request}
				}
				if c := atomic.AddInt64(&processed, 1); c%5000 == 0 {
					rate := float64(c) / time.Since(t0).Seconds()
					log.Printf("processed %d/%d (%.0f rps)", c, n, rate)
				}
			}
			doneWorkers <- struct{}{}
		}()
	}
	for w := 0; w < wgN; w++ {
		<-doneWorkers
	}
	close(mm)
	<-done
	close(stop)

	log.Printf("DONE in %v: %d mismatches", time.Since(t0).Round(time.Millisecond), len(mismatches))
	for _, m := range mismatches {
		fmt.Printf("--- entry %d: expected=%.1f got=%.1f ---\n%s\n", m.idx, m.expected, m.got, m.req)
	}
}
