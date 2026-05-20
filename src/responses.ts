// Six pre-baked HTTP response payloads, indexed by fraud count (0..5).
// fraud_score = count / 5, approved = count < 3 (since 0.6 = 3/5 is denied).

const BODIES: string[] = [
  '{"approved":true,"fraud_score":0.0}',
  '{"approved":true,"fraud_score":0.2}',
  '{"approved":true,"fraud_score":0.4}',
  '{"approved":false,"fraud_score":0.6}',
  '{"approved":false,"fraud_score":0.8}',
  '{"approved":false,"fraud_score":1.0}',
];

export const BODY_BUFFERS: Buffer[] = BODIES.map((s) => Buffer.from(s, 'utf8'));

export const HTTP_RESPONSES: Buffer[] = BODIES.map((body) =>
  Buffer.from(
    `HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${Buffer.byteLength(body)}\r\nConnection: keep-alive\r\n\r\n${body}`,
    'utf8',
  ),
);

export const READY_RESPONSE = Buffer.from(
  'HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: keep-alive\r\n\r\nready',
  'utf8',
);

export const NOT_READY_RESPONSE = Buffer.from(
  'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 5\r\nConnection: keep-alive\r\n\r\nbusy.',
  'utf8',
);

export const NOT_FOUND_RESPONSE = Buffer.from(
  'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n',
  'utf8',
);
