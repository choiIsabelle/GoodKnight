import { NextRequest, NextResponse } from "next/server";
import { ensureServerRunning } from "../server";
import http from "http";

export const runtime = "nodejs";

const servePort = Number(process.env.SERVE_PORT || "5058");

async function proxyToLocal(req: NextRequest) {
  await ensureServerRunning();

  const body = await req.text();

  return new Promise<NextResponse>((resolve, reject) => {
    const proxyReq = http.request(
      {
        host: "127.0.0.1",
        port: servePort,
        path: "/move",
        method: "POST",
        headers: {
          ...req.headers,
          host: `127.0.0.1:${servePort}`,
        },
      },
      (proxyRes) => {
        let responseBody = "";
        proxyRes.on("data", (chunk) => {
          responseBody += chunk;
        });
        proxyRes.on("end", () => {
          const headers = new Headers();
          Object.keys(proxyRes.headers).forEach((key) => {
            const value = proxyRes.headers[key];
            if (value) {
              headers.set(key, Array.isArray(value) ? value.join(", ") : value);
            }
          });
          resolve(
            new NextResponse(responseBody, {
              status: proxyRes.statusCode,
              headers,
            })
          );
        });
      }
    );

    proxyReq.on("error", (err) => {
      console.error("Failed to proxy to serve.py", err);
      reject(err);
    });

    proxyReq.write(body);
    proxyReq.end();
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

  console.log("Proxying request to backend /move", backendUrl);
  const response = await fetch(`${backendUrl}/move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  console.log("Received response from serve.py /move", {
    status: response.status,
  });
  const contentType =
    response.headers.get("content-type") || "application/json";

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
