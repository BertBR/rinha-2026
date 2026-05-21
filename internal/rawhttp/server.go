// Package rawhttp is a tiny HTTP/1.1 server tuned for the Rinha 2026
// fraud-score schema. It bypasses net/http to keep the hot path allocation-
// and copy-free: pooled read buffers, a header scanner specialized for the
// two endpoints we actually serve, and prebaked response byte slices.

package rawhttp

import (
	"io"
	"net"
	"strconv"
	"sync"
	"time"
)

// Handler is the contract the api server fulfills.
type Handler interface {
	ServeFraudScore(body []byte) []byte
	ServeReady() []byte
}

var (
	fraudResponses [6][]byte
	readyResponse  []byte
	notFound       []byte
)

func init() {
	bodies := [6]string{
		`{"approved":true,"fraud_score":0.0}`,
		`{"approved":true,"fraud_score":0.2}`,
		`{"approved":true,"fraud_score":0.4}`,
		`{"approved":false,"fraud_score":0.6}`,
		`{"approved":false,"fraud_score":0.8}`,
		`{"approved":false,"fraud_score":1.0}`,
	}
	for i, b := range bodies {
		fraudResponses[i] = buildOK("application/json", b)
	}
	readyResponse = buildOK("", "OK")
	notFound = buildResp(404, "Not Found", "", "")
}

func buildOK(ct, body string) []byte {
	return buildResp(200, "OK", ct, body)
}

func buildResp(code int, text, ct, body string) []byte {
	hdr := "HTTP/1.1 " + strconv.Itoa(code) + " " + text + "\r\n"
	if ct != "" {
		hdr += "Content-Type: " + ct + "\r\n"
	}
	hdr += "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n"
	out := make([]byte, len(hdr)+len(body))
	copy(out, hdr)
	copy(out[len(hdr):], body)
	return out
}

const (
	readBufSize    = 2048
	maxRequestSize = 4 * 1024
)

var bufPool = sync.Pool{
	New: func() any { b := make([]byte, readBufSize); return &b },
}

// Server is the rawhttp dispatcher. One instance is shared by every
// per-FD goroutine spawned from the SCM_RIGHTS control loop.
type Server struct {
	handler Handler
}

func New(h Handler) *Server { return &Server{handler: h} }

// ServeConn runs the per-connection request loop until EOF or error.
// The caller owns the conn lifecycle.
func (s *Server) ServeConn(conn net.Conn) {
	defer conn.Close()

	bp := bufPool.Get().(*[]byte)
	defer bufPool.Put(bp)
	buf := *bp
	pos, used := 0, 0

	for {
		if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
			return
		}

		headEnd := indexHeaderEnd(buf[pos:used])
		for headEnd < 0 {
			if used == len(buf) {
				return // request larger than our buffer
			}
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
				if used-pos > maxRequestSize {
					return
				}
				headEnd = indexHeaderEnd(buf[pos:used])
				continue
			}
			if err != nil {
				if err == io.EOF && headEnd >= 0 {
					break
				}
				return
			}
		}
		headEnd += pos + 4 // include trailing \r\n\r\n

		method, path, contentLen := parseRequestLine(buf[pos:headEnd])

		bodyEnd := headEnd + contentLen
		for used < bodyEnd {
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
				continue
			}
			if err != nil {
				return
			}
		}

		var resp []byte
		switch {
		// POST /fraud-score — fixed-length path, fast tag check
		case len(method) == 4 && method[0] == 'P' && len(path) == 12 && path[1] == 'f':
			resp = s.handler.ServeFraudScore(buf[headEnd:bodyEnd])
		// GET /ready
		case len(method) == 3 && method[0] == 'G' && len(path) == 6 && path[1] == 'r':
			resp = s.handler.ServeReady()
		default:
			resp = notFound
		}

		if err := conn.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
			return
		}
		if _, err := conn.Write(resp); err != nil {
			return
		}

		pos = bodyEnd
		if pos >= used {
			pos, used = 0, 0
		}
	}
}

func indexHeaderEnd(b []byte) int {
	for i := 0; i+3 < len(b); i++ {
		if b[i] == '\r' && b[i+1] == '\n' && b[i+2] == '\r' && b[i+3] == '\n' {
			return i
		}
	}
	return -1
}

func parseRequestLine(buf []byte) (method, path []byte, contentLen int) {
	i := 0
	for i < len(buf) && buf[i] != ' ' {
		i++
	}
	method = buf[:i]
	i++
	start := i
	for i < len(buf) && buf[i] != ' ' {
		i++
	}
	path = buf[start:i]
	contentLen = findContentLength(buf)
	return
}

func findContentLength(buf []byte) int {
	for i := 0; i+16 < len(buf); i++ {
		if (buf[i] == 'C' || buf[i] == 'c') && isContentLength(buf[i:]) {
			j := i + 16
			for j < len(buf) && buf[j] == ' ' {
				j++
			}
			n := 0
			for j < len(buf) && buf[j] >= '0' && buf[j] <= '9' {
				n = n*10 + int(buf[j]-'0')
				j++
			}
			return n
		}
	}
	return 0
}

func isContentLength(b []byte) bool {
	const prefix = "content-length:"
	if len(b) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		c := b[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != prefix[i] {
			return false
		}
	}
	return true
}

func FraudResponse(count int) []byte {
	if count < 0 || count >= len(fraudResponses) {
		return fraudResponses[0]
	}
	return fraudResponses[count]
}

func ReadyResponse() []byte { return readyResponse }
