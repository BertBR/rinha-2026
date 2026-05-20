// Hand-rolled JSON parser specialized for the Rinha 2026 fraud-score payload.
//
// The payload schema is fixed and the field order is stable across all test
// data we have seen. We exploit both: instead of building a JS object tree
// (which JSON.parse does at the cost of ~12 GC-able allocations per request),
// we scan bytes once and write the 14 required values directly into a
// pre-allocated Float32Array.
//
// At 900 rps this saves about 10K allocations/second and the associated young-
// gen GC pressure that landed on our p99 tail.
//
// Field order assumed (from API.md and observed payloads):
//   id, transaction{amount,installments,requested_at}, customer{avg_amount,
//   tx_count_24h, known_merchants}, merchant{id,mcc,avg_amount},
//   terminal{is_online,card_present,km_from_home},
//   last_transaction(null | {timestamp,km_from_current})

import { DIMS } from './quantize.ts';
import type { Normalization, MccRiskMap } from './vector.ts';

const CH_QUOTE = 0x22;
const CH_COLON = 0x3a;
const CH_COMMA = 0x2c;
const CH_LBRACE = 0x7b;
const CH_RBRACE = 0x7d;
const CH_LBRACK = 0x5b;
const CH_RBRACK = 0x5d;
const CH_MINUS = 0x2d;
const CH_DOT = 0x2e;
const CH_T = 0x74; // 't' for true
const CH_N = 0x6e; // 'n' for null
const CH_0 = 0x30;
const CH_9 = 0x39;
const CH_E_LOWER = 0x65;
const CH_E_UPPER = 0x45;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

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

// Advance past whitespace/control bytes.
function skipWs(buf: Buffer, pos: number, end: number): number {
  while (pos < end) {
    const c = buf[pos];
    if (c > 32 && c !== 0x09 && c !== 0x0a && c !== 0x0d) return pos;
    pos++;
  }
  return pos;
}

// Advance to the next `:`, skipping the field name (which we don't validate;
// we assume canonical field order).
function skipToColon(buf: Buffer, pos: number, end: number): number {
  while (pos < end && buf[pos] !== CH_COLON) pos++;
  return pos + 1;
}

// Skip a quoted string value (handles backslash escapes minimally).
function skipString(buf: Buffer, pos: number, end: number): number {
  // expects to land on opening quote
  pos = skipWs(buf, pos, end);
  pos++; // opening "
  while (pos < end) {
    const c = buf[pos];
    if (c === 0x5c) {
      pos += 2;
      continue;
    }
    if (c === CH_QUOTE) return pos + 1;
    pos++;
  }
  return pos;
}

// Read a JSON number until a delimiter; returns the parsed float and new pos.
// Returns NaN on parse failure.
function readNumber(buf: Buffer, pos: number, end: number): [number, number] {
  pos = skipWs(buf, pos, end);
  let neg = false;
  if (buf[pos] === CH_MINUS) {
    neg = true;
    pos++;
  }
  let intPart = 0;
  while (pos < end) {
    const c = buf[pos];
    if (c < CH_0 || c > CH_9) break;
    intPart = intPart * 10 + (c - CH_0);
    pos++;
  }
  let value = intPart;
  if (buf[pos] === CH_DOT) {
    pos++;
    let fracPart = 0;
    let fracScale = 1;
    while (pos < end) {
      const c = buf[pos];
      if (c < CH_0 || c > CH_9) break;
      fracPart = fracPart * 10 + (c - CH_0);
      fracScale *= 10;
      pos++;
    }
    value += fracPart / fracScale;
  }
  if (buf[pos] === CH_E_LOWER || buf[pos] === CH_E_UPPER) {
    pos++;
    let expNeg = false;
    if (buf[pos] === CH_MINUS) {
      expNeg = true;
      pos++;
    } else if (buf[pos] === 0x2b) {
      pos++;
    }
    let exp = 0;
    while (pos < end) {
      const c = buf[pos];
      if (c < CH_0 || c > CH_9) break;
      exp = exp * 10 + (c - CH_0);
      pos++;
    }
    value *= Math.pow(10, expNeg ? -exp : exp);
  }
  return [neg ? -value : value, pos];
}

// Read an int (no fractional part) until a delimiter.
function readInt(buf: Buffer, pos: number, end: number): [number, number] {
  pos = skipWs(buf, pos, end);
  let neg = false;
  if (buf[pos] === CH_MINUS) {
    neg = true;
    pos++;
  }
  let value = 0;
  while (pos < end) {
    const c = buf[pos];
    if (c < CH_0 || c > CH_9) break;
    value = value * 10 + (c - CH_0);
    pos++;
  }
  return [neg ? -value : value, pos];
}

// Read a 2-digit ASCII number at fixed offset; no advance.
function read2(buf: Buffer, off: number): number {
  return (buf[off] - CH_0) * 10 + (buf[off + 1] - CH_0);
}

function read4(buf: Buffer, off: number): number {
  return (
    (buf[off] - CH_0) * 1000 +
    (buf[off + 1] - CH_0) * 100 +
    (buf[off + 2] - CH_0) * 10 +
    (buf[off + 3] - CH_0)
  );
}

export interface ParseContext {
  norm: Normalization;
  mccRisk: MccRiskMap;
  // Pre-allocated working buffers; reused across requests.
  merchantIdStart: number;
  merchantIdLen: number;
  // Up to N known_merchants positions captured during parse.
  knownStarts: Int32Array;
  knownLens: Int32Array;
  knownCount: number;
}

export function makeParseContext(
  norm: Normalization,
  mccRisk: MccRiskMap,
): ParseContext {
  return {
    norm,
    mccRisk,
    merchantIdStart: 0,
    merchantIdLen: 0,
    knownStarts: new Int32Array(64),
    knownLens: new Int32Array(64),
    knownCount: 0,
  };
}

function bytesEqual(
  buf: Buffer,
  aStart: number,
  aLen: number,
  bStart: number,
  bLen: number,
): boolean {
  if (aLen !== bLen) return false;
  for (let i = 0; i < aLen; i++) {
    if (buf[aStart + i] !== buf[bStart + i]) return false;
  }
  return true;
}

// Parse the payload bytes and write the 14-dim vector into `out`.
// Returns true on success, false on parse failure (caller should respond with
// the safe fallback).
export function parseAndVectorize(
  buf: Buffer,
  end: number,
  out: Float32Array,
  ctx: ParseContext,
): boolean {
  const norm = ctx.norm;
  const mccRisk = ctx.mccRisk;
  let pos = 0;

  // Outer object
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACE) return false;
  pos++;

  // "id" : "<string>"
  pos = skipToColon(buf, pos, end);
  pos = skipString(buf, pos, end);

  // ,"transaction": { ... }
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACE) return false;
  pos++;

  // amount
  pos = skipToColon(buf, pos, end);
  const [amount, p1] = readNumber(buf, pos, end);
  pos = p1;
  // installments
  pos = skipToColon(buf, pos, end);
  const [installments, p2] = readInt(buf, pos, end);
  pos = p2;
  // requested_at: capture string positions
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  pos++; // opening "
  const tsReqStart = pos;
  // ISO format: YYYY-MM-DDTHH:MM:SSZ = 20 chars
  const reqYear = read4(buf, pos);
  const reqMonth = read2(buf, pos + 5);
  const reqDay = read2(buf, pos + 8);
  const reqHour = read2(buf, pos + 11);
  const reqMinute = read2(buf, pos + 14);
  const reqSecond = read2(buf, pos + 17);
  // advance past the string (find closing ")
  while (pos < end && buf[pos] !== CH_QUOTE) pos++;
  pos++;

  // closing transaction object: },"customer":
  pos = skipWs(buf, pos, end);
  if (buf[pos] === CH_RBRACE) pos++;

  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACE) return false;
  pos++;

  // avg_amount
  pos = skipToColon(buf, pos, end);
  const [avgAmount, p3] = readNumber(buf, pos, end);
  pos = p3;
  // tx_count_24h
  pos = skipToColon(buf, pos, end);
  const [txCount24h, p4] = readInt(buf, pos, end);
  pos = p4;
  // known_merchants: [ "MERC-001", "MERC-002", ... ]
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACK) return false;
  pos++;
  ctx.knownCount = 0;
  while (pos < end) {
    pos = skipWs(buf, pos, end);
    if (buf[pos] === CH_RBRACK) {
      pos++;
      break;
    }
    if (buf[pos] === CH_COMMA) {
      pos++;
      continue;
    }
    if (buf[pos] !== CH_QUOTE) return false;
    pos++;
    const sStart = pos;
    while (pos < end && buf[pos] !== CH_QUOTE) pos++;
    const sLen = pos - sStart;
    if (ctx.knownCount < ctx.knownStarts.length) {
      ctx.knownStarts[ctx.knownCount] = sStart;
      ctx.knownLens[ctx.knownCount] = sLen;
      ctx.knownCount++;
    }
    pos++; // closing "
  }

  // closing customer: },"merchant":{
  pos = skipWs(buf, pos, end);
  if (buf[pos] === CH_RBRACE) pos++;

  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACE) return false;
  pos++;

  // merchant.id - capture position
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  pos++; // opening "
  ctx.merchantIdStart = pos;
  while (pos < end && buf[pos] !== CH_QUOTE) pos++;
  ctx.merchantIdLen = pos - ctx.merchantIdStart;
  pos++;

  // merchant.mcc - capture digits, parse as integer
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  pos++; // opening "
  let mccInt = 0;
  while (pos < end && buf[pos] !== CH_QUOTE) {
    const c = buf[pos];
    if (c >= CH_0 && c <= CH_9) {
      mccInt = mccInt * 10 + (c - CH_0);
    }
    pos++;
  }
  pos++;

  // merchant.avg_amount
  pos = skipToColon(buf, pos, end);
  const [merchantAvgAmount, p5] = readNumber(buf, pos, end);
  pos = p5;

  // closing merchant: },"terminal":{
  pos = skipWs(buf, pos, end);
  if (buf[pos] === CH_RBRACE) pos++;

  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  if (buf[pos] !== CH_LBRACE) return false;
  pos++;

  // is_online
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  const isOnline = buf[pos] === CH_T ? 1 : 0;
  while (pos < end && buf[pos] !== CH_COMMA && buf[pos] !== CH_RBRACE) pos++;

  // card_present
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);
  const cardPresent = buf[pos] === CH_T ? 1 : 0;
  while (pos < end && buf[pos] !== CH_COMMA && buf[pos] !== CH_RBRACE) pos++;

  // km_from_home
  pos = skipToColon(buf, pos, end);
  const [kmFromHome, p6] = readNumber(buf, pos, end);
  pos = p6;

  // closing terminal: },"last_transaction":
  pos = skipWs(buf, pos, end);
  if (buf[pos] === CH_RBRACE) pos++;
  pos = skipToColon(buf, pos, end);
  pos = skipWs(buf, pos, end);

  let lastMinutes = -1;
  let lastKm = -1;
  if (buf[pos] === CH_N) {
    // null
    while (pos < end && buf[pos] !== CH_COMMA && buf[pos] !== CH_RBRACE) pos++;
  } else if (buf[pos] === CH_LBRACE) {
    pos++;
    // timestamp
    pos = skipToColon(buf, pos, end);
    pos = skipWs(buf, pos, end);
    pos++; // opening "
    const lastYear = read4(buf, pos);
    const lastMonth = read2(buf, pos + 5);
    const lastDay = read2(buf, pos + 8);
    const lastHour = read2(buf, pos + 11);
    const lastMinute = read2(buf, pos + 14);
    const lastSecond = read2(buf, pos + 17);
    while (pos < end && buf[pos] !== CH_QUOTE) pos++;
    pos++;

    const reqSec =
      daysSinceEpoch(reqYear, reqMonth, reqDay) * 86400 +
      reqHour * 3600 +
      reqMinute * 60 +
      reqSecond;
    const lastSec =
      daysSinceEpoch(lastYear, lastMonth, lastDay) * 86400 +
      lastHour * 3600 +
      lastMinute * 60 +
      lastSecond;
    lastMinutes = (reqSec - lastSec) / 60;

    // km_from_current
    pos = skipToColon(buf, pos, end);
    const [k, p7] = readNumber(buf, pos, end);
    pos = p7;
    lastKm = k;
  } else {
    return false;
  }

  // Compute vector
  out[0] = clamp01(amount / norm.max_amount);
  out[1] = clamp01(installments / norm.max_installments);
  out[2] = clamp01(amount / avgAmount / norm.amount_vs_avg_ratio);
  out[3] = reqHour / 23;
  const dse = daysSinceEpoch(reqYear, reqMonth, reqDay);
  const dow = (((dse + 3) % 7) + 7) % 7;
  out[4] = dow / 6;
  if (lastMinutes < 0) {
    out[5] = -1;
    out[6] = -1;
  } else {
    out[5] = clamp01(lastMinutes / norm.max_minutes);
    out[6] = clamp01(lastKm / norm.max_km);
  }
  out[7] = clamp01(kmFromHome / norm.max_km);
  out[8] = clamp01(txCount24h / norm.max_tx_count_24h);
  out[9] = isOnline;
  out[10] = cardPresent;

  // unknown_merchant: 1 if merchant.id not in known_merchants
  let known = 0;
  for (let i = 0; i < ctx.knownCount; i++) {
    if (
      bytesEqual(
        buf,
        ctx.knownStarts[i],
        ctx.knownLens[i],
        ctx.merchantIdStart,
        ctx.merchantIdLen,
      )
    ) {
      known = 1;
      break;
    }
  }
  out[11] = 1 - known;

  // mcc_risk lookup. JSON keys are strings. Use String of mccInt as key.
  // (One allocation per request — small string, transient.) An alternative
  // is to pre-build a fixed-size Uint8Array keyed by the integer; the keys
  // we have observed are all 4-digit MCC codes.
  const risk = mccRisk[String(mccInt)];
  out[12] = risk === undefined ? 0.5 : risk;

  out[13] = clamp01(merchantAvgAmount / norm.max_merchant_avg_amount);

  // Reference variables we computed but didn't use (intentionally suppressed)
  void tsReqStart;
  void reqSecond;
  void reqMinute;

  return true;
}
