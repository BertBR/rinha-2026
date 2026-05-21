#!/usr/bin/env bash
# Local load test against docker-compose up.
# Runs k6 (or autocannon) at 900rps for 30s and dumps p99.
#
# Usage: ./scripts/bench.sh [duration-s]
set -euo pipefail

DUR=${1:-30}
URL=http://localhost:9999/fraud-score

# Pick a representative payload from data/example-payloads.json.
PAYLOAD=$(node -e "console.log(JSON.stringify(JSON.parse(require('fs').readFileSync('data/example-payloads.json'))[0]))")

if command -v k6 >/dev/null 2>&1; then
  echo "[bench] using k6 against $URL"
  cat > /tmp/bench.js <<JS
import http from 'k6/http';
import { check } from 'k6';

const payload = $PAYLOAD;

export const options = {
  scenarios: {
    constant: {
      executor: 'constant-arrival-rate',
      rate: 900,
      timeUnit: '1s',
      duration: '${DUR}s',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(95)', 'p(99)', 'p(99.9)', 'max'],
};

export default function () {
  const res = http.post('${URL}', JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(res, { '200': (r) => r.status === 200 });
}
JS
  k6 run /tmp/bench.js
elif command -v autocannon >/dev/null 2>&1; then
  echo "[bench] using autocannon against $URL"
  autocannon -c 50 -d $DUR -m POST -H 'Content-Type: application/json' -b "$PAYLOAD" "$URL"
else
  echo "ERROR: install k6 or autocannon"
  exit 1
fi
