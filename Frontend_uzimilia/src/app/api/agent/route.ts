// src/app/api/agent/route.ts
import type { NextRequest } from 'next/server';

export async function GET(request: NextRequest) {
  // recrea la URL de tu API remota usando los mismos parámetros
  const url = `@url del agente en cloud run?` +
    new URL(request.url).searchParams.toString();

  // forward
  const apiRes = await fetch(url);
  const text   = await apiRes.text();

  return new Response(text, {
    status: apiRes.status,
    headers: { 'Content-Type': 'text/plain' },
  });
}
