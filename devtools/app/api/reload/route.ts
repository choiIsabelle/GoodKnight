// Placeholder /api/reload for static export. For dev, run the Python server locally
// and use NEXT_PUBLIC_REMOTE_BACKEND_URL in the frontend to call the backend.
export const dynamic = "force-static";
import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json({ reloaded: false, message: "Reload disabled in static export" });
}
