// Minimal TCP -> UDS reverse proxy. No HTTP parsing; pure byte forwarding.
//
// Round-robin across N upstream Unix domain sockets at the *connection* level.
// With HTTP keep-alive (which uWS + k6 both use), a single TCP connection
// serves many sequential requests, so connection-level round-robin spreads
// load across upstreams roughly evenly when there are >> 2 VUs (k6 here uses
// up to 250).
//
// Why not HTTP-aware load balancing? Because we don't have to be:
//   - All upstreams are equivalent (no path-based routing)
//   - Per-request round-robin requires parsing HTTP request boundaries,
//     adding ~200-500 µs per request and a parser bug surface.
// HAProxy in `mode http` does the parsing version. This binary skips it.
//
// Env:
//   PORT       — TCP port to listen on (default 9999)
//   UPSTREAMS  — comma-separated UDS paths (default /sockets/api1.sock,/sockets/api2.sock)
//   WORKERS    — tokio worker threads (default 2)

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::copy_bidirectional;
use tokio::net::{TcpListener, UnixStream};

fn main() {
    let workers: usize = env::var("WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async_main());
}

async fn async_main() {
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);
    let upstreams_str = env::var("UPSTREAMS")
        .unwrap_or_else(|_| "/sockets/api1.sock,/sockets/api2.sock".to_string());
    let upstreams: Arc<Vec<String>> = Arc::new(
        upstreams_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    );
    if upstreams.is_empty() {
        eprintln!("[lb] no upstreams configured");
        std::process::exit(1);
    }

    eprintln!(
        "[lb] listening on 0.0.0.0:{port}, upstreams={:?}, workers={}",
        upstreams,
        env::var("WORKERS").unwrap_or_else(|_| "2".to_string())
    );

    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("bind tcp");

    let counter = Arc::new(AtomicUsize::new(0));

    loop {
        let (mut client, _addr) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                eprintln!("[lb] accept error: {e}");
                continue;
            }
        };
        let _ = client.set_nodelay(true);

        let upstreams = Arc::clone(&upstreams);
        let counter = Arc::clone(&counter);

        tokio::spawn(async move {
            let n = upstreams.len();
            let idx = counter.fetch_add(1, Ordering::Relaxed) % n;

            // Try the chosen upstream first; fall through to the next on
            // connect failure. Avoids a single bad sock taking down the LB.
            let mut connected: Option<UnixStream> = None;
            for try_idx in 0..n {
                let path = &upstreams[(idx + try_idx) % n];
                match UnixStream::connect(path).await {
                    Ok(s) => {
                        connected = Some(s);
                        break;
                    }
                    Err(_) => continue,
                }
            }

            let mut upstream = match connected {
                Some(s) => s,
                None => return,
            };

            let _ = copy_bidirectional(&mut client, &mut upstream).await;
        });
    }
}
