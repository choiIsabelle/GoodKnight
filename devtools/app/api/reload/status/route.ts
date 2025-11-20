// Static /api/reload/status placeholder for export builds.
export const dynamic = "force-static";
import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({ lastRestartAt: null, lastRestartReason: null });
}
