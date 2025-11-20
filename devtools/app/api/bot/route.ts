// Static placeholder for /api/bot when building a static export.
// In export mode the frontend should call a remote backend set via
// NEXT_PUBLIC_REMOTE_BACKEND_URL; this route is intentionally a
// no-op to keep the site exportable.
export const dynamic = "force-static";
import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json(
    {
      error: "Local API proxy disabled in static export. Set NEXT_PUBLIC_REMOTE_BACKEND_URL to your backend.",
    },
    { status: 501 }
  );
}
