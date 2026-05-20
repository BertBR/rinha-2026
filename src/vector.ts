// Payload → 14-dim float32 vector. Follows DETECTION_RULES.md exactly.
//
// Timestamp parsing is character-code arithmetic on the fixed ISO-8601 layout
// "YYYY-MM-DDThh:mm:ssZ" — no Date construction, no string allocation. Day of
// week uses days-since-epoch (Jan 1 1970 was Thursday).

import { DIMS } from './quantize.ts';

const MIN_CHAR = 48;

export interface Normalization {
  max_amount: number;
  max_installments: number;
  amount_vs_avg_ratio: number;
  max_minutes: number;
  max_km: number;
  max_tx_count_24h: number;
  max_merchant_avg_amount: number;
}

export type MccRiskMap = Record<string, number>;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function readUint2(s: string, off: number): number {
  return (s.charCodeAt(off) - MIN_CHAR) * 10 + (s.charCodeAt(off + 1) - MIN_CHAR);
}

function readUint4(s: string, off: number): number {
  return (
    (s.charCodeAt(off) - MIN_CHAR) * 1000 +
    (s.charCodeAt(off + 1) - MIN_CHAR) * 100 +
    (s.charCodeAt(off + 2) - MIN_CHAR) * 10 +
    (s.charCodeAt(off + 3) - MIN_CHAR)
  );
}

// Days since 1970-01-01 for a UTC date. Avoids Date.UTC allocation.
function daysSinceEpoch(year: number, month: number, day: number): number {
  let y = year;
  let m = month;
  if (m <= 2) {
    y -= 1;
    m += 12;
  }
  const era = Math.floor(y / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (m - 3) + 2) / 5) + day - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

// Seconds since 1970-01-01 00:00:00 UTC for a parsed ISO timestamp.
function unixSeconds(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number,
): number {
  return daysSinceEpoch(year, month, day) * 86400 + hour * 3600 + minute * 60 + second;
}

export interface VectorizeContext {
  norm: Normalization;
  mccRisk: MccRiskMap;
}

// Writes 14 floats into `out`. `payload` is the request body parsed via JSON.parse.
export function vectorize(payload: any, out: Float32Array, ctx: VectorizeContext): void {
  const norm = ctx.norm;
  const mccRisk = ctx.mccRisk;

  const tx = payload.transaction;
  const cust = payload.customer;
  const merch = payload.merchant;
  const term = payload.terminal;
  const lt = payload.last_transaction;

  const ts: string = tx.requested_at;

  // 0
  out[0] = clamp01(tx.amount / norm.max_amount);
  // 1
  out[1] = clamp01(tx.installments / norm.max_installments);
  // 2
  out[2] = clamp01(tx.amount / cust.avg_amount / norm.amount_vs_avg_ratio);

  // 3 + 4: hour and day_of_week from ts
  const year = readUint4(ts, 0);
  const month = readUint2(ts, 5);
  const day = readUint2(ts, 8);
  const hour = readUint2(ts, 11);
  const minute = readUint2(ts, 14);
  const second = readUint2(ts, 17);

  out[3] = hour / 23;
  const dse = daysSinceEpoch(year, month, day);
  // Jan 1 1970 was Thursday. We want mon=0, sun=6.
  // Thursday = 3 in our scheme. dse=0 → 3. So dow = (dse + 3) mod 7.
  const dow = (((dse + 3) % 7) + 7) % 7;
  out[4] = dow / 6;

  // 5 + 6: last_transaction
  if (lt !== null && lt !== undefined) {
    const lts: string = lt.timestamp;
    const lyear = readUint4(lts, 0);
    const lmonth = readUint2(lts, 5);
    const lday = readUint2(lts, 8);
    const lhour = readUint2(lts, 11);
    const lminute = readUint2(lts, 14);
    const lsecond = readUint2(lts, 17);
    const lastSec = unixSeconds(lyear, lmonth, lday, lhour, lminute, lsecond);
    const currSec = unixSeconds(year, month, day, hour, minute, second);
    const minutes = (currSec - lastSec) / 60;
    out[5] = clamp01(minutes / norm.max_minutes);
    out[6] = clamp01(lt.km_from_current / norm.max_km);
  } else {
    out[5] = -1;
    out[6] = -1;
  }

  // 7
  out[7] = clamp01(term.km_from_home / norm.max_km);
  // 8
  out[8] = clamp01(cust.tx_count_24h / norm.max_tx_count_24h);
  // 9
  out[9] = term.is_online ? 1 : 0;
  // 10
  out[10] = term.card_present ? 1 : 0;
  // 11: unknown_merchant — 1 if merchant.id NOT in known_merchants
  const mid = merch.id;
  const km = cust.known_merchants;
  let known = 0;
  for (let i = 0; i < km.length; i++) {
    if (km[i] === mid) {
      known = 1;
      break;
    }
  }
  out[11] = 1 - known;
  // 12
  const risk = mccRisk[merch.mcc];
  out[12] = risk === undefined ? 0.5 : risk;
  // 13
  out[13] = clamp01(merch.avg_amount / norm.max_merchant_avg_amount);
}

export function emptyVec(): Float32Array {
  return new Float32Array(DIMS);
}
