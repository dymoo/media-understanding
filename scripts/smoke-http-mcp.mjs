#!/usr/bin/env node

const baseUrl = process.env["MCP_BASE_URL"] ?? "http://127.0.0.1:8000";
const endpoint = new URL("/mcp", baseUrl).toString();
const mediaPath = process.env["MCP_SMOKE_MEDIA_PATH"];
const protocolVersion = "2024-11-05";

function fail(message) {
  throw new Error(message);
}

async function postJson(body, sessionId) {
  const headers = {
    accept: "application/json, text/event-stream",
    "content-type": "application/json",
  };
  if (sessionId) headers["mcp-session-id"] = sessionId;

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  const text = await response.text();
  const payload = text.startsWith("event:")
    ? text
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n")
    : text;

  let json;
  try {
    json = payload ? JSON.parse(payload) : null;
  } catch {
    fail(`Non-JSON response from ${body.method}: ${text}`);
  }

  return { response, json };
}

async function waitForHealth() {
  const healthUrl = new URL("/healthz", baseUrl).toString();
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      const response = await fetch(healthUrl);
      if (response.ok) return;
    } catch {
      // Keep polling until the server starts.
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  fail(`Timed out waiting for health endpoint: ${healthUrl}`);
}

await waitForHealth();

const init = await postJson({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion,
    capabilities: {},
    clientInfo: { name: "docker-smoke", version: "1.0.0" },
  },
});

if (!init.response.ok) fail(`initialize failed with HTTP ${init.response.status}`);

const sessionId = init.response.headers.get("mcp-session-id");
if (!sessionId) fail("initialize response missing mcp-session-id header");

if (!init.json?.result) fail(`initialize returned invalid payload: ${JSON.stringify(init.json)}`);

const toolsList = await postJson(
  {
    jsonrpc: "2.0",
    id: 2,
    method: "tools/list",
    params: {},
  },
  sessionId,
);

if (!toolsList.response.ok) fail(`tools/list failed with HTTP ${toolsList.response.status}`);

const tools = toolsList.json?.result?.tools;
if (!Array.isArray(tools))
  fail(`tools/list returned invalid payload: ${JSON.stringify(toolsList.json)}`);

const toolNames = tools.map((tool) => tool?.name).filter(Boolean);
for (const required of [
  "probe_media",
  "understand_media",
  "get_transcript",
  "get_video_grids",
  "get_frames",
]) {
  if (!toolNames.includes(required)) fail(`Missing expected tool: ${required}`);
}

if (mediaPath) {
  const probe = await postJson(
    {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "probe_media",
        arguments: {
          paths: mediaPath,
        },
      },
    },
    sessionId,
  );

  if (!probe.response.ok) fail(`probe_media failed with HTTP ${probe.response.status}`);
  if (probe.json?.error) fail(`probe_media returned error: ${JSON.stringify(probe.json.error)}`);
}

console.log(`ok: ${endpoint}`);
