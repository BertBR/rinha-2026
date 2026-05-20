// Local k6 harness — mirrors the upstream profile so local results map to the
// official scoring. Pulls test data straight from the upstream test file once
// at startup.
//
// Usage:
//   make data              # downloads test-data.json into ../data/
//   k6 run bench/k6.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

const BASE = __ENV.BASE_URL || 'http://localhost:9999';

const payloads = new SharedArray('payloads', () => {
  const data = JSON.parse(open('../data/test-data.json'));
  return data.entries || data;
});

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: 1,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 250,
      gracefulStop: '10s',
      stages: [{ target: 900, duration: '120s' }],
    },
  },
  thresholds: {
    http_req_duration: ['p(99)<5'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const i = Math.floor(Math.random() * payloads.length);
  const entry = payloads[i];
  const payload = entry.request || entry.payload || entry;
  const res = http.post(`${BASE}/fraud-score`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '2001ms',
  });
  check(res, {
    '200 ok': (r) => r.status === 200,
  });
}

export function setup() {
  // Block until /ready is 200 (server still warming up).
  const start = Date.now();
  while (Date.now() - start < 60_000) {
    const r = http.get(`${BASE}/ready`);
    if (r.status === 200) return;
    sleep(1);
  }
  throw new Error('server never became ready');
}
