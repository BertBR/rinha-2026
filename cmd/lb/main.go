// Tiny TCP-to-FD load balancer for Rinha 2026.
//
// Listens on TCP :9999. For each accepted connection, picks a worker round-
// robin and hands the raw client fd over a Unix STREAM control socket using
// SCM_RIGHTS ancillary data. The LB then closes its own end; the kernel keeps
// the underlying socket alive because the worker received an independent
// dup'd fd. No bytes are proxied through this process — userspace HAProxy
// hop disappears, p99 drops by 4-5 ms on the rinha hardware.

package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

var (
	listenAddr  = flag.String("listen", ":9999", "TCP listen address")
	workerAddrs = flag.String("workers", "/sockets/api1.ctrl,/sockets/api2.ctrl", "comma-separated worker control socket paths")
)

func main() {
	flag.Parse()

	paths := strings.Split(*workerAddrs, ",")
	cons := make([]*net.UnixConn, 0, len(paths))
	// Workers may not be ready yet (they bind their ctrl socket before
	// finishing kNN init). Retry a few times before giving up.
	for _, p := range paths {
		p = strings.TrimSpace(p)
		var conn *net.UnixConn
		for attempt := 0; attempt < 60; attempt++ {
			c, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: p, Net: "unix"})
			if err == nil {
				conn = c
				break
			}
			time.Sleep(500 * time.Millisecond)
		}
		if conn == nil {
			log.Fatalf("[lb] dial worker %s: gave up after 30s", p)
		}
		cons = append(cons, conn)
	}
	log.Printf("[lb] connected to %d workers", len(cons))

	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("[lb] listen %s: %v", *listenAddr, err)
	}
	log.Printf("[lb] listening on %s", *listenAddr)

	var rr uint64
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		ln.Close()
		for _, c := range cons {
			c.Close()
		}
	}()

	for {
		tcpConn, err := ln.Accept()
		if err != nil {
			return
		}
		wi := atomic.AddUint64(&rr, 1) % uint64(len(cons))
		raw, err := tcpConn.(*net.TCPConn).SyscallConn()
		if err != nil {
			tcpConn.Close()
			continue
		}
		var fd int
		if err := raw.Control(func(f uintptr) { fd = int(f) }); err != nil {
			tcpConn.Close()
			continue
		}
		rights := syscall.UnixRights(fd)
		_ = cons[wi].SetWriteDeadline(time.Now().Add(50 * time.Millisecond))
		if _, _, err = cons[wi].WriteMsgUnix(nil, rights, nil); err != nil {
			log.Printf("[lb] write to worker %d: %v", wi, err)
		}
		tcpConn.Close() // the worker holds its own dup'd fd
	}
}
