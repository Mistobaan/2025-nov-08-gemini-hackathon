import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { code, description } = await req.json();

    if (typeof code !== "string" || typeof description !== "string") {
      return NextResponse.json({ error: "Code and description are required." }, { status: 400 });
    }

    const hasSolutionStub = code.includes("def solution");
    const review = [
      "TPU Review:",
      "Use the `solution` function as your TPU entry point and keep the implementation TPU-friendly (vectorize ops, minimize host/device transfers).",
      `Problem description:\n${description}`,
      hasSolutionStub
        ? "Great, the `solution` stub is in place—wire it up with TPU-ready inputs and return tensors."
        : "Define a `solution` function so TPU workloads know where to execute your logic.",
      "When you flesh this out, rely on JAX primitives or TPU-optimized libraries instead of Python loops."
    ].join("\n\n");

    return NextResponse.json({ review });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to create TPU review.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
