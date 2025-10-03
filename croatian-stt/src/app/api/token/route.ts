import { NextResponse } from "next/server";

type JwtHeader = {
  alg: "HS256";
  typ: "JWT";
};

type JwtPayload = {
  sub: string;
  iat: number;
  exp: number;
};

const encoder = new TextEncoder();

function base64UrlFromArray(array: Uint8Array) {
  return Buffer.from(array)
    .toString("base64")
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

async function signHs256(content: string, secret: string) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(content),
  );
  return base64UrlFromArray(new Uint8Array(signature));
}

export async function GET() {
  const secret = process.env.AUTH_SECRET;
  if (!secret) {
    return NextResponse.json({ error: "Missing AUTH_SECRET" }, { status: 500 });
  }

  const header: JwtHeader = { alg: "HS256", typ: "JWT" };
  const issuedAt = Math.floor(Date.now() / 1000);
  const payload: JwtPayload = {
    sub: "stt-client",
    iat: issuedAt,
    exp: issuedAt + 600,
  };

  const headerSegment = base64UrlFromArray(
    encoder.encode(JSON.stringify(header)),
  );
  const payloadSegment = base64UrlFromArray(
    encoder.encode(JSON.stringify(payload)),
  );
  const signingInput = `${headerSegment}.${payloadSegment}`;
  const signatureSegment = await signHs256(signingInput, secret);
  const token = `${signingInput}.${signatureSegment}`;

  return NextResponse.json({ token });
}
