// API worker for Rinha 2026.
//
// The api does not bind a TCP socket of its own. Instead it listens on a
// Unix STREAM control socket and waits for the companion LB to hand it
// already-accepted client connections via SCM_RIGHTS file-descriptor passing.
//
// Per request:
//   1. Wait for an incoming FD over CTRL_SOCKET.
//   2. Reconstruct it as a net.Conn (the kernel ref-counts; closing in the
//      LB does not affect us).
//   3. Spawn a goroutine running the rawhttp dispatcher; it parses HTTP/1.1
//      directly on the conn, calls the kNN core, writes one of the six
//      prebaked JSON responses, and returns when the client closes.

package main

import (
	"encoding/json"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/BertBR/rinha-2026/internal/knn"
	"github.com/BertBR/rinha-2026/internal/rawhttp"
	"github.com/BertBR/rinha-2026/internal/vec"
)

type reqState struct {
	q   [14]float32
	ctx *vec.ParseContext
}

var statePool sync.Pool

var (
	normalization *vec.Normalization
	mccRisk       vec.MccRiskMap

	ready   atomic.Bool
	readyCh = make(chan struct{})
)

type fraudHandler struct{}

func (fraudHandler) ServeFraudScore(body []byte) []byte {
	// Block first request(s) on init. Subsequent requests fly through.
	if !ready.Load() {
		<-readyCh
	}
	st := statePool.Get().(*reqState)
	defer statePool.Put(st)

	var count uint8
	if vec.ParseAndVectorize(body, &st.q, st.ctx) {
		count = knn.FraudCount(&st.q)
	}
	return rawhttp.FraudResponse(int(count))
}

func (fraudHandler) ServeReady() []byte {
	return rawhttp.ReadyResponse()
}

func main() {
	dataDir := envOr("DATA_DIR", "./data")
	ctrlSocket := envOr("CTRL_SOCKET", "/sockets/api.ctrl")

	if err := loadConfig(dataDir); err != nil {
		log.Fatalf("[api] load config: %v", err)
	}
	statePool.New = func() any {
		return &reqState{ctx: vec.NewContext(normalization, mccRisk)}
	}

	// Bind the control socket BEFORE init so the LB's connect succeeds.
	_ = os.Remove(ctrlSocket)
	ctrlLn, err := net.Listen("unix", ctrlSocket)
	if err != nil {
		log.Fatalf("[api] listen ctrl %s: %v", ctrlSocket, err)
	}
	if err := os.Chmod(ctrlSocket, 0o666); err != nil {
		log.Printf("[api] chmod ctrl: %v", err)
	}

	go func() {
		log.Printf("[api] initializing native kNN")
		t0 := time.Now()
		knn.Init()
		log.Printf("[api]   ready in %v", time.Since(t0).Round(time.Millisecond))
		ready.Store(true)
		close(readyCh)
	}()

	srv := rawhttp.New(fraudHandler{})

	log.Printf("[api] ctrl socket %s (GOMAXPROCS=%d)", ctrlSocket, runtime.GOMAXPROCS(0))
	for {
		ctrlConn, err := ctrlLn.Accept()
		if err != nil {
			log.Printf("[api] ctrl accept: %v", err)
			return
		}
		go serveControl(ctrlConn, srv)
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

// serveControl reads SCM_RIGHTS messages from the LB; each one carries a
// dup'd TCP fd we can wrap in a net.Conn and serve directly.
func serveControl(ctrlConn net.Conn, srv *rawhttp.Server) {
	defer ctrlConn.Close()
	uc, ok := ctrlConn.(*net.UnixConn)
	if !ok {
		return
	}
	buf := make([]byte, 1)
	oob := make([]byte, 64)
	for {
		_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}
		fd, err := parseUnixRights(oob[:oobn])
		if err != nil || fd < 0 {
			continue
		}
		file := os.NewFile(uintptr(fd), "")
		conn, err := net.FileConn(file)
		_ = file.Close()
		if err != nil {
			continue
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetNoDelay(true)
		}
		go srv.ServeConn(conn)
	}
}

// parseUnixRights extracts the single fd from a SCM_RIGHTS cmsg.
// Avoids syscall.ParseSocketControlMessage which allocates.
func parseUnixRights(oob []byte) (int, error) {
	if len(oob) < 20 {
		return -1, nil
	}
	level := int(oob[8]) | int(oob[9])<<8 | int(oob[10])<<16 | int(oob[11])<<24
	typ := int(oob[12]) | int(oob[13])<<8 | int(oob[14])<<16 | int(oob[15])<<24
	if level != syscall.SOL_SOCKET || typ != syscall.SCM_RIGHTS {
		return -1, nil
	}
	fd := int(oob[16]) | int(oob[17])<<8 | int(oob[18])<<16 | int(oob[19])<<24
	return fd, nil
}
