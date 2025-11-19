import { NextRequest, NextResponse } from "next/server";
import { ensureServerRunning } from "../server";

export const runtime = "nodejs";

const servePort = Number(process.env.SERVE_PORT || "5058");

async function proxyToLocal(req: NextRequest) {
  // ensure local Python server is running (dev only)
  await ensureServerRunning();

  const body = await req.text();

  const res = await fetch(`http://127.0.0.1:${servePort}/move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });

  const text = await res.text();
  const contentType = res.headers.get("content-type") || "application/json";

  return new NextResponse(text, {
    status: res.status,
    headers: {
      "Content-Type": contentType,
    },
  });
}

async function proxyToRemote(req: NextRequest, backendUrl: string) {
  let body: any;
  try {
    body = await req.json();
  } catch (err) {
    console.error("/api/bot invalid JSON", err);
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const response = await fetch(`${backendUrl}/move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  const contentType = response.headers.get("content-type") || "application/json";

  return new NextResponse(text, {
    status: response.status,
    headers: {
      "Content-Type": contentType,
    },
  });
}

export async function POST(req: NextRequest) {
  console.log("/api/bot POST received");

  const backendUrl = process.env.NEXT_PUBLIC_REMOTE_BACKEND_URL;

  if (backendUrl) {
    return proxyToRemote(req, backendUrl);
  } else {
    return proxyToLocal(req);
  }
}
