// Package vec parses the fixed Rinha 2026 fraud-score payload and emits a
// 14-dimensional float32 vector ready for the kNN core. The parser is
// hand-rolled because the schema is stable and we need to avoid the GC
// pressure that comes with encoding/json on the hot path.
//
// Field order assumed (canonical):
//   id, transaction{amount,installments,requested_at},
//   customer{avg_amount,tx_count_24h,known_merchants},
//   merchant{id,mcc,avg_amount},
//   terminal{is_online,card_present,km_from_home},
//   last_transaction(null | {timestamp,km_from_current})

package vec

import (
	"math"
	"strconv"
)

const DIMS = 14

type Normalization struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

type MccRiskMap map[string]float64

const (
	chQuote  = 0x22
	chColon  = 0x3a
	chComma  = 0x2c
	chLBrace = 0x7b
	chRBrace = 0x7d
	chLBrack = 0x5b
	chRBrack = 0x5d
	chMinus  = 0x2d
	chPlus   = 0x2b
	chDot    = 0x2e
	chT      = 0x74
	chN      = 0x6e
	ch0      = 0x30
	ch9      = 0x39
	chELo    = 0x65
	chEUp    = 0x45
)

func clamp01(v float64) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return float32(v)
}

// daysSinceEpoch: days from 1970-01-01 to the given UTC date.
func daysSinceEpoch(year, month, day int) int {
	y, m := year, month
	if m <= 2 {
		y -= 1
		m += 12
	}
	era := y / 400
	if y < 0 && y%400 != 0 {
		era -= 1
	}
	yoe := y - era*400
	doy := (153*(m-3)+2)/5 + day - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}

func skipWs(b []byte, pos, end int) int {
	for pos < end {
		c := b[pos]
		if c > 32 && c != 0x09 && c != 0x0a && c != 0x0d {
			return pos
		}
		pos++
	}
	return pos
}

func skipToColon(b []byte, pos, end int) int {
	for pos < end && b[pos] != chColon {
		pos++
	}
	return pos + 1
}

func skipString(b []byte, pos, end int) int {
	pos = skipWs(b, pos, end)
	pos++ // opening "
	for pos < end {
		c := b[pos]
		if c == 0x5c {
			pos += 2
			continue
		}
		if c == chQuote {
			return pos + 1
		}
		pos++
	}
	return pos
}

func readNumber(b []byte, pos, end int) (float64, int) {
	pos = skipWs(b, pos, end)
	neg := false
	if pos < end && b[pos] == chMinus {
		neg = true
		pos++
	}
	intPart := 0.0
	for pos < end {
		c := b[pos]
		if c < ch0 || c > ch9 {
			break
		}
		intPart = intPart*10 + float64(c-ch0)
		pos++
	}
	value := intPart
	if pos < end && b[pos] == chDot {
		pos++
		fracPart := 0.0
		fracScale := 1.0
		for pos < end {
			c := b[pos]
			if c < ch0 || c > ch9 {
				break
			}
			fracPart = fracPart*10 + float64(c-ch0)
			fracScale *= 10
			pos++
		}
		value += fracPart / fracScale
	}
	if pos < end && (b[pos] == chELo || b[pos] == chEUp) {
		pos++
		expNeg := false
		if pos < end {
			if b[pos] == chMinus {
				expNeg = true
				pos++
			} else if b[pos] == chPlus {
				pos++
			}
		}
		exp := 0
		for pos < end {
			c := b[pos]
			if c < ch0 || c > ch9 {
				break
			}
			exp = exp*10 + int(c-ch0)
			pos++
		}
		if expNeg {
			exp = -exp
		}
		value *= math.Pow(10, float64(exp))
	}
	if neg {
		value = -value
	}
	return value, pos
}

func readInt(b []byte, pos, end int) (int, int) {
	pos = skipWs(b, pos, end)
	neg := false
	if pos < end && b[pos] == chMinus {
		neg = true
		pos++
	}
	value := 0
	for pos < end {
		c := b[pos]
		if c < ch0 || c > ch9 {
			break
		}
		value = value*10 + int(c-ch0)
		pos++
	}
	if neg {
		value = -value
	}
	return value, pos
}

func read2(b []byte, off int) int {
	return int(b[off]-ch0)*10 + int(b[off+1]-ch0)
}

func read4(b []byte, off int) int {
	return int(b[off]-ch0)*1000 +
		int(b[off+1]-ch0)*100 +
		int(b[off+2]-ch0)*10 +
		int(b[off+3]-ch0)
}

// ParseContext holds reusable per-request scratch state. Caller owns one
// per goroutine.
type ParseContext struct {
	Norm    *Normalization
	MccRisk MccRiskMap
	// Captured positions for merchant.id and known_merchants entries.
	merchantIDStart int
	merchantIDLen   int
	knownStarts     [64]int
	knownLens       [64]int
	knownCount      int
}

func NewContext(norm *Normalization, mcc MccRiskMap) *ParseContext {
	return &ParseContext{Norm: norm, MccRisk: mcc}
}

func bytesEqual(b []byte, aStart, aLen, bStart, bLen int) bool {
	if aLen != bLen {
		return false
	}
	for i := 0; i < aLen; i++ {
		if b[aStart+i] != b[bStart+i] {
			return false
		}
	}
	return true
}

// ParseAndVectorize scans the payload bytes and writes 14 floats into out.
// Returns true on success, false on parse error.
func ParseAndVectorize(buf []byte, out *[14]float32, ctx *ParseContext) bool {
	norm := ctx.Norm
	mcc := ctx.MccRisk
	end := len(buf)
	pos := 0

	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrace {
		return false
	}
	pos++

	// "id": "<string>"
	pos = skipToColon(buf, pos, end)
	pos = skipString(buf, pos, end)

	// ,"transaction": { ... }
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrace {
		return false
	}
	pos++

	pos = skipToColon(buf, pos, end)
	amount, p1 := readNumber(buf, pos, end)
	pos = p1

	pos = skipToColon(buf, pos, end)
	installments, p2 := readInt(buf, pos, end)
	pos = p2

	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	pos++ // opening "
	if pos+19 >= end {
		return false
	}
	reqYear := read4(buf, pos)
	reqMonth := read2(buf, pos+5)
	reqDay := read2(buf, pos+8)
	reqHour := read2(buf, pos+11)
	reqMinute := read2(buf, pos+14)
	reqSecond := read2(buf, pos+17)
	for pos < end && buf[pos] != chQuote {
		pos++
	}
	pos++

	pos = skipWs(buf, pos, end)
	if pos < end && buf[pos] == chRBrace {
		pos++
	}

	// ,"customer":{
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrace {
		return false
	}
	pos++

	pos = skipToColon(buf, pos, end)
	avgAmount, p3 := readNumber(buf, pos, end)
	pos = p3

	pos = skipToColon(buf, pos, end)
	txCount24h, p4 := readInt(buf, pos, end)
	pos = p4

	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrack {
		return false
	}
	pos++
	ctx.knownCount = 0
	for pos < end {
		pos = skipWs(buf, pos, end)
		if pos >= end {
			return false
		}
		if buf[pos] == chRBrack {
			pos++
			break
		}
		if buf[pos] == chComma {
			pos++
			continue
		}
		if buf[pos] != chQuote {
			return false
		}
		pos++
		sStart := pos
		for pos < end && buf[pos] != chQuote {
			pos++
		}
		sLen := pos - sStart
		if ctx.knownCount < len(ctx.knownStarts) {
			ctx.knownStarts[ctx.knownCount] = sStart
			ctx.knownLens[ctx.knownCount] = sLen
			ctx.knownCount++
		}
		pos++ // closing "
	}

	pos = skipWs(buf, pos, end)
	if pos < end && buf[pos] == chRBrace {
		pos++
	}

	// ,"merchant":{
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrace {
		return false
	}
	pos++

	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	pos++ // opening "
	ctx.merchantIDStart = pos
	for pos < end && buf[pos] != chQuote {
		pos++
	}
	ctx.merchantIDLen = pos - ctx.merchantIDStart
	pos++

	// merchant.mcc — capture digits as int
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	pos++ // opening "
	mccInt := 0
	for pos < end && buf[pos] != chQuote {
		c := buf[pos]
		if c >= ch0 && c <= ch9 {
			mccInt = mccInt*10 + int(c-ch0)
		}
		pos++
	}
	pos++

	pos = skipToColon(buf, pos, end)
	merchantAvgAmount, p5 := readNumber(buf, pos, end)
	pos = p5

	pos = skipWs(buf, pos, end)
	if pos < end && buf[pos] == chRBrace {
		pos++
	}

	// ,"terminal":{
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	if pos >= end || buf[pos] != chLBrace {
		return false
	}
	pos++

	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	isOnline := float32(0)
	if pos < end && buf[pos] == chT {
		isOnline = 1
	}
	for pos < end && buf[pos] != chComma && buf[pos] != chRBrace {
		pos++
	}

	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)
	cardPresent := float32(0)
	if pos < end && buf[pos] == chT {
		cardPresent = 1
	}
	for pos < end && buf[pos] != chComma && buf[pos] != chRBrace {
		pos++
	}

	pos = skipToColon(buf, pos, end)
	kmFromHome, p6 := readNumber(buf, pos, end)
	pos = p6

	pos = skipWs(buf, pos, end)
	if pos < end && buf[pos] == chRBrace {
		pos++
	}
	pos = skipToColon(buf, pos, end)
	pos = skipWs(buf, pos, end)

	lastMinutes := -1.0
	lastKm := -1.0
	if pos < end && buf[pos] == chN {
		for pos < end && buf[pos] != chComma && buf[pos] != chRBrace {
			pos++
		}
	} else if pos < end && buf[pos] == chLBrace {
		pos++
		pos = skipToColon(buf, pos, end)
		pos = skipWs(buf, pos, end)
		pos++ // opening "
		if pos+19 >= end {
			return false
		}
		lastYear := read4(buf, pos)
		lastMonth := read2(buf, pos+5)
		lastDay := read2(buf, pos+8)
		lastHour := read2(buf, pos+11)
		lastMinute := read2(buf, pos+14)
		lastSecond := read2(buf, pos+17)
		for pos < end && buf[pos] != chQuote {
			pos++
		}
		pos++

		reqSec := daysSinceEpoch(reqYear, reqMonth, reqDay)*86400 +
			reqHour*3600 + reqMinute*60 + reqSecond
		lastSec := daysSinceEpoch(lastYear, lastMonth, lastDay)*86400 +
			lastHour*3600 + lastMinute*60 + lastSecond
		lastMinutes = float64(reqSec-lastSec) / 60.0

		pos = skipToColon(buf, pos, end)
		k, p7 := readNumber(buf, pos, end)
		pos = p7
		lastKm = k
	} else {
		return false
	}
	_ = pos // last position unused after this point

	out[0] = clamp01(amount / norm.MaxAmount)
	out[1] = clamp01(float64(installments) / norm.MaxInstallments)
	out[2] = clamp01(amount / avgAmount / norm.AmountVsAvgRatio)
	out[3] = float32(reqHour) / 23
	dse := daysSinceEpoch(reqYear, reqMonth, reqDay)
	dow := ((dse+3)%7 + 7) % 7
	out[4] = float32(dow) / 6
	if lastMinutes < 0 {
		out[5] = -1
		out[6] = -1
	} else {
		out[5] = clamp01(lastMinutes / norm.MaxMinutes)
		out[6] = clamp01(lastKm / norm.MaxKm)
	}
	out[7] = clamp01(kmFromHome / norm.MaxKm)
	out[8] = clamp01(float64(txCount24h) / norm.MaxTxCount24h)
	out[9] = isOnline
	out[10] = cardPresent

	known := 0
	for i := 0; i < ctx.knownCount; i++ {
		if bytesEqual(buf, ctx.knownStarts[i], ctx.knownLens[i], ctx.merchantIDStart, ctx.merchantIDLen) {
			known = 1
			break
		}
	}
	out[11] = float32(1 - known)

	if risk, ok := mcc[strconv.Itoa(mccInt)]; ok {
		out[12] = float32(risk)
	} else {
		out[12] = 0.5
	}

	out[13] = clamp01(merchantAvgAmount / norm.MaxMerchantAvgAmount)

	return true
}
