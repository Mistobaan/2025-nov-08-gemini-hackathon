import { NextRequest, NextResponse } from "next/server";
import { stackServerApp } from "./stack/server";

export async function middleware(request: NextRequest) {
    const pathname = request.nextUrl.pathname;

    // Define paths that don't require authentication
    const publicPaths = ["/", "/favicon.ico", "/robots.txt"];

    // Check if the path is public or a system path (like _next)
    if (
        publicPaths.includes(pathname) ||
        pathname.startsWith("/_next") ||
        pathname.startsWith("/handler") || // Stack Auth handler paths
        pathname.startsWith("/api/stack") // Stack Auth API paths if any
    ) {
        return NextResponse.next();
    }

    // Check for authenticated user
    const user = await stackServerApp.getUser();

    if (!user) {
        // Redirect to sign-in if not authenticated
        // We use the handler path for sign-in
        const url = request.nextUrl.clone();
        url.pathname = "/handler/sign-in";
        // Add return_to parameter to redirect back after login
        url.searchParams.set("return_to", pathname);
        return NextResponse.redirect(url);
    }

    return NextResponse.next();
}

export const config = {
    matcher: [
        /*
         * Match all request paths except for the ones starting with:
         * - api (API routes)
         * - _next/static (static files)
         * - _next/image (image optimization files)
         * - favicon.ico (favicon file)
         */
        '/((?!api|_next/static|_next/image|favicon.ico).*)',
    ],
};
