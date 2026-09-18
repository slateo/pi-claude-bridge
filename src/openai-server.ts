import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { MODEL_IDS_IN_ORDER } from "./models.js";
import { configureStandaloneBridge, streamClaudeAgentSdk } from "./index.js";
import {
	chatRequestToRelay,
	ContinuationRegistry,
	rawSystemPrompt,
	type ChatMessage,
	type ChatRequest,
	type ContinuationLease,
} from "./openai-relay.js";
type Usage = { input?: number; output?: number; cacheRead?: number; cacheWrite?: number };
type HarnessUsageWindow = { utilization: number; resets_at: string | null };
type HarnessUsagePayload = {
	five_hour?: HarnessUsageWindow;
	seven_day?: HarnessUsageWindow;
	limits?: Array<{
		kind: "weekly_scoped";
		percent: number;
		resets_at: string | null;
		is_active: true;
		scope: { model: { display_name: string } };
	}>;
};

const DEFAULT_MAX_BODY = 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 30 * 60_000;
const DEFAULT_HEARTBEAT_MS = 15_000;
const PI_SYSTEM_PROMPT_START = "You are an expert coding assistant operating inside pi, a coding agent harness.";
const PI_SYSTEM_PROMPT_END = "- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)";
const modelSet = new Set(MODEL_IDS_IN_ORDER);

const USAGE_LINES = [
	{ pattern: /^Current session:\s+([\d.]+)% used\s+·\s+resets\s+(.+)$/m, key: "five_hour" },
	{ pattern: /^Current week \(all models\):\s+([\d.]+)% used\s+·\s+resets\s+(.+)$/m, key: "seven_day" },
	{ pattern: /^Current week \(Fable\):\s+([\d.]+)% used\s+·\s+resets\s+(.+)$/m, key: "fable" },
] as const;

function zonedDateParts(date: Date, timeZone: string): Record<string, number> {
	const parts = new Intl.DateTimeFormat("en-US", {
		timeZone,
		year: "numeric",
		month: "2-digit",
		day: "2-digit",
		hour: "2-digit",
		minute: "2-digit",
		second: "2-digit",
		hourCycle: "h23",
	}).formatToParts(date);
	return Object.fromEntries(parts.filter((part) => part.type !== "literal").map((part) => [part.type, Number(part.value)]));
}

/** Convert Claude Code's local `/usage` reset label into an ISO instant. */
export function parseUsageResetAt(label: string, now = new Date()): string | null {
	const match = /^([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{1,2})(?::(\d{2}))?(am|pm)\s+\(([^)]+)\)$/.exec(label.trim());
	if (!match) return null;
	const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
	const month = months.indexOf(match[1]);
	if (month < 0) return null;
	const timeZone = match[6];
	let hour = Number(match[3]) % 12;
	if (match[5] === "pm") hour += 12;
	const minute = Number(match[4] || 0);
	const day = Number(match[2]);
	try {
		const current = zonedDateParts(now, timeZone);
		let year = current.year;
		const candidateDay = Date.UTC(year, month, day);
		const currentDay = Date.UTC(current.year, current.month - 1, current.day);
		if (candidateDay < currentDay - 180 * 24 * 60 * 60 * 1000) year += 1;

		const wallClock = Date.UTC(year, month, day, hour, minute, 0);
		let instant = wallClock;
		for (let iteration = 0; iteration < 2; iteration += 1) {
			const observed = zonedDateParts(new Date(instant), timeZone);
			const observedWallClock = Date.UTC(
				observed.year,
				observed.month - 1,
				observed.day,
				observed.hour,
				observed.minute,
				observed.second,
			);
			instant += wallClock - observedWallClock;
		}
		return new Date(instant).toISOString();
	} catch {
		return null;
	}
}

export function parseHarnessUsage(text: string, now = new Date()): HarnessUsagePayload {
	const usage: HarnessUsagePayload = {};
	for (const line of USAGE_LINES) {
		const match = line.pattern.exec(text);
		if (!match) continue;
		const percent = Number(match[1]);
		if (!Number.isFinite(percent)) continue;
		const resetsAt = parseUsageResetAt(match[2], now);
		if (line.key === "five_hour" || line.key === "seven_day") {
			usage[line.key] = { utilization: percent, resets_at: resetsAt };
		} else {
			usage.limits = [{
				kind: "weekly_scoped",
				percent,
				resets_at: resetsAt,
				is_active: true,
				scope: { model: { display_name: "Fable" } },
			}];
		}
	}
	return usage;
}

function textContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((part) => {
			if (!part || typeof part !== "object") return "";
			const record = part as Record<string, unknown>;
			return (record.type === "text" || record.type === "input_text") && typeof record.text === "string"
				? record.text
				: "";
		})
		.filter(Boolean)
		.join("\n");
}

/** Preserve the caller Pi session's working directory for host-local tool use. */
export function piClientWorkingDirectory(messages: ChatMessage[], fallback: string): string {
	const system = messages
		.filter((message) => message.role === "system" || message.role === "developer")
		.map((message) => textContent(message.content))
		.join("\n\n");
	const matches = [...system.matchAll(/(?:^|\n)Current working directory:\s*([^\n]+)\s*$/gm)];
	const candidate = matches.at(-1)?.[1]?.trim();
	return candidate && isAbsolute(candidate) ? candidate : fallback;
}

/**
 * Remove Pi's generated harness wrapper before handing a request to the inner
 * Claude Code harness. Portable instructions appended after that wrapper (for
 * example project context and skills) are retained. Unknown prompt layouts are
 * returned unchanged so a Pi upgrade cannot silently discard user instructions.
 */
export function projectPiClientSystemPrompt(system: string): string {
	if (!system.startsWith(PI_SYSTEM_PROMPT_START)) return system;
	const wrapperEnd = system.indexOf(PI_SYSTEM_PROMPT_END);
	if (wrapperEnd < 0) return system;

	let portable = system.slice(wrapperEnd + PI_SYSTEM_PROMPT_END.length).trim();
	portable = portable.replace(/(?:^|\n\n)Current working directory: [^\n]+\s*$/, "").trim();
	return portable;
}

function json(res: ServerResponse, status: number, value: unknown): void {
	const body = JSON.stringify(value);
	res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
	res.end(body);
}

function authorized(req: IncomingMessage, apiKey: string): boolean {
	return req.headers.authorization === `Bearer ${apiKey}`;
}

async function readBody(req: IncomingMessage, maxBytes: number): Promise<unknown> {
	const chunks: Buffer[] = [];
	let size = 0;
	for await (const chunk of req) {
		const buffer = Buffer.from(chunk);
		size += buffer.length;
		if (size > maxBytes) throw new Error("request body exceeds limit");
		chunks.push(buffer);
	}
	return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function runClaude(args: string[], timeoutMs = 30_000): Promise<string> {
	const claudeBin = process.env.CLAUDE_BRIDGE_CLAUDE_BIN || "claude";
	const child = spawn(claudeBin, args, {
		cwd: process.env.CLAUDE_BRIDGE_CWD || "/tmp",
		env: { ...process.env },
		stdio: ["ignore", "pipe", "pipe"],
	});
	let stdout = "";
	let stderr = "";
	const collect = (current: string, chunk: Buffer): string => (current + chunk.toString()).slice(-256 * 1024);
	child.stdout.on("data", (chunk) => { stdout = collect(stdout, chunk); });
	child.stderr.on("data", (chunk) => { stderr = collect(stderr, chunk); });
	const timeout = setTimeout(() => child.kill("SIGTERM"), timeoutMs);
	const exitCode = await new Promise<number | null>((resolveExit, reject) => {
		child.once("error", reject);
		child.once("close", resolveExit);
	});
	clearTimeout(timeout);
	if (exitCode !== 0) {
		throw new Error(stderr.trim().split("\n").slice(-1)[0] || `Claude Code exited with code ${exitCode}`);
	}
	return stdout;
}

async function getHarnessUsage(): Promise<unknown> {
	const [authText, usageText] = await Promise.all([
		runClaude(["auth", "status", "--json"]),
		runClaude(["-p", "--output-format", "json", "/usage"]),
	]);
	const auth = JSON.parse(authText) as { loggedIn?: boolean; authMethod?: string; subscriptionType?: string };
	const usageResult = JSON.parse(usageText) as { result?: string };
	if (!auth.loggedIn || auth.authMethod !== "claude.ai") {
		throw new Error("Claude Code harness is not logged in with a Claude subscription");
	}
	const usage = parseHarnessUsage(String(usageResult.result || ""));
	if (!usage.five_hour && !usage.seven_day && !usage.limits?.length) {
		throw new Error("Claude Code /usage returned no recognizable quota windows");
	}
	const subscriptionType = String(auth.subscriptionType || "").toLowerCase();
	return {
		id: process.env.CLAUDE_BRIDGE_HARNESS_ID || "default",
		name: process.env.CLAUDE_BRIDGE_HARNESS_NAME || "Claude Code Harness",
		usage,
		profile: {
			account: {
				has_claude_max: subscriptionType === "max",
				has_claude_pro: subscriptionType === "pro",
			},
		},
		subscription_type: subscriptionType || null,
		observed_at: new Date().toISOString(),
	};
}

function usageShape(usage: Usage | undefined) {
	return {
		prompt_tokens: (usage?.input || 0) + (usage?.cacheRead || 0) + (usage?.cacheWrite || 0),
		completion_tokens: usage?.output || 0,
		total_tokens: (usage?.input || 0) + (usage?.cacheRead || 0) + (usage?.cacheWrite || 0) + (usage?.output || 0),
	};
}

function completionChunk(
	id: string,
	created: number,
	model: string,
	delta: Record<string, unknown>,
	finishReason: "stop" | "length" | "tool_calls" | null = null,
	usage?: ReturnType<typeof usageShape>,
): string {
	return `data: ${JSON.stringify({
		id,
		object: "chat.completion.chunk",
		created,
		model,
		choices: [{ index: 0, delta, finish_reason: finishReason }],
		...(usage ? { usage } : {}),
	})}\n\n`;
}

function writeStreamChunk(res: ServerResponse, chunk: string): boolean {
	if (res.destroyed || res.writableEnded) return false;
	res.write(chunk);
	return true;
}

async function handleChat(
	res: ServerResponse,
	body: ChatRequest,
	continuations: ContinuationRegistry,
	streamFn: typeof streamClaudeAgentSdk,
): Promise<void> {
	const modelId = String(body.model || "");
	if (!modelSet.has(modelId)) return json(res, 400, { error: { message: `unsupported model: ${modelId}` } });
	if (!Array.isArray(body.messages) || body.messages.length === 0) {
		return json(res, 400, { error: { message: "messages must be a non-empty array" } });
	}

	const system = projectPiClientSystemPrompt(rawSystemPrompt(body));
	const relay = chatRequestToRelay(body, system);
	const lease: ContinuationLease = relay.resultIds.length > 0
		? continuations.resume(modelId, relay.resultIds)
		: continuations.create(modelId);
	const id = `chatcmpl-${randomUUID()}`;
	const created = Math.floor(Date.now() / 1000);
	const timeoutMs = Number(process.env.CLAUDE_BRIDGE_TIMEOUT_MS || DEFAULT_TIMEOUT_MS);
	let timedOut = false;
	const timeout = setTimeout(() => {
		timedOut = true;
		continuations.cancel(lease, `Claude relay timed out after ${timeoutMs}ms`);
	}, timeoutMs);
	timeout.unref();

	let heartbeat: NodeJS.Timeout | undefined;
	let responseEnded = false;
	const onClientClose = () => {
		if (!responseEnded && !res.writableEnded) continuations.cancel(lease, "Claude relay client disconnected");
	};
	res.once("close", onClientClose);

	if (body.stream) {
		res.writeHead(200, {
			"content-type": "text/event-stream",
			"cache-control": "no-cache, no-transform",
			connection: "keep-alive",
			"x-accel-buffering": "no",
		});
		writeStreamChunk(res, completionChunk(id, created, modelId, { role: "assistant" }));
		const heartbeatMs = Number(process.env.CLAUDE_BRIDGE_HEARTBEAT_MS || DEFAULT_HEARTBEAT_MS);
		heartbeat = setInterval(() => writeStreamChunk(res, completionChunk(id, created, modelId, {})), heartbeatMs);
		heartbeat.unref();
	}

	let finalMessage: import("@earendil-works/pi-ai").AssistantMessage | undefined;
	let terminalError: string | undefined;
	try {
		const stream = streamFn(relay.model, relay.context, {
			reasoning: relay.reasoning,
			signal: lease.controller.signal,
			cwd: piClientWorkingDirectory(body.messages, process.env.CLAUDE_BRIDGE_CWD || "/tmp"),
			metadata: { claudeBridgeIsolatedSession: true },
		} as import("@earendil-works/pi-ai").SimpleStreamOptions & { cwd: string });

		for await (const event of stream) {
			if (event.type === "thinking_delta" && body.stream) {
				writeStreamChunk(res, completionChunk(id, created, modelId, { reasoning_content: event.delta }));
			} else if (event.type === "text_delta" && body.stream) {
				writeStreamChunk(res, completionChunk(id, created, modelId, { content: event.delta }));
			} else if (event.type === "toolcall_end" && body.stream) {
				const call = event.toolCall;
				writeStreamChunk(res, completionChunk(id, created, modelId, {
					tool_calls: [{
						index: event.partial.content.filter((block) => block.type === "toolCall").findIndex((block) => block.id === call.id),
						id: call.id,
						type: "function",
						function: { name: call.name, arguments: JSON.stringify(call.arguments ?? {}) },
					}],
				}));
			} else if (event.type === "done") {
				finalMessage = event.message;
			} else if (event.type === "error") {
				finalMessage = event.error;
				terminalError = event.error.errorMessage || `Claude relay ${event.reason}`;
			}
		}
	} catch (error) {
		terminalError = timedOut
			? `Claude relay timed out after ${timeoutMs}ms`
			: error instanceof Error ? error.message : String(error);
	} finally {
		clearTimeout(timeout);
		if (heartbeat) clearInterval(heartbeat);
		res.off("close", onClientClose);
	}

	if (res.destroyed) {
		continuations.cancel(lease, terminalError || "Claude relay response destroyed");
		return;
	}
	if (terminalError || !finalMessage) {
		continuations.cancel(lease, terminalError || "Claude relay ended without a final message");
		const message = terminalError || "Claude relay ended without a final message";
		if (body.stream) {
			writeStreamChunk(res, `data: ${JSON.stringify({ error: { message } })}\n\n`);
			responseEnded = true;
			res.end("data: [DONE]\n\n");
			return;
		}
		responseEnded = true;
		return json(res, 502, { error: { message } });
	}

	const toolCalls = finalMessage.content.filter((block) => block.type === "toolCall");
	const finishReason = finalMessage.stopReason === "toolUse"
		? "tool_calls"
		: finalMessage.stopReason === "length" ? "length" : "stop";
	if (finishReason === "tool_calls") {
		if (toolCalls.length === 0) {
			continuations.cancel(lease, "Claude relay stopped for tools without tool calls");
			throw new Error("Claude relay stopped for tools without tool calls");
		}
		continuations.bind(lease, toolCalls.map((call) => call.id));
	} else {
		continuations.finish(lease);
	}

	const usage = usageShape(finalMessage.usage);
	if (body.stream) {
		writeStreamChunk(res, completionChunk(id, created, modelId, {}, finishReason, usage));
		responseEnded = true;
		res.end("data: [DONE]\n\n");
		return;
	}

	const text = finalMessage.content
		.filter((block) => block.type === "text")
		.map((block) => block.text)
		.join("");
	responseEnded = true;
	json(res, 200, {
		id,
		object: "chat.completion",
		created,
		model: modelId,
		choices: [{
			index: 0,
			message: {
				role: "assistant",
				content: text || null,
				...(toolCalls.length > 0 ? {
					tool_calls: toolCalls.map((call) => ({
						id: call.id,
						type: "function",
						function: { name: call.name, arguments: JSON.stringify(call.arguments ?? {}) },
					})),
				} : {}),
			},
			finish_reason: finishReason,
		}],
		usage,
	});
}

export function startServer(options: { streamFn?: typeof streamClaudeAgentSdk } = {}) {
	const apiKey = process.env.CLAUDE_BRIDGE_API_KEY;
	if (!apiKey) throw new Error("CLAUDE_BRIDGE_API_KEY is required");
	const host = process.env.CLAUDE_BRIDGE_HOST || "127.0.0.1";
	const port = Number(process.env.CLAUDE_BRIDGE_PORT || 8318);
	const maxBody = Number(process.env.CLAUDE_BRIDGE_MAX_BODY_BYTES || DEFAULT_MAX_BODY);
	const continuationTtlMs = Number(process.env.CLAUDE_BRIDGE_CONTINUATION_TTL_MS || DEFAULT_TIMEOUT_MS);
	const streamFn = options.streamFn ?? streamClaudeAgentSdk;
	configureStandaloneBridge(process.env.CLAUDE_BRIDGE_CWD || "/tmp");
	const continuations = new ContinuationRegistry(continuationTtlMs);
	const server = createServer(async (req, res) => {
		try {
			if (!authorized(req, apiKey)) return json(res, 401, { error: { message: "unauthorized" } });
			const path = new URL(req.url || "/", "http://localhost").pathname;
			if (req.method === "GET" && path === "/health") return json(res, 200, { status: "ok" });
			if (req.method === "GET" && path === "/v1/models") {
				return json(res, 200, { object: "list", data: MODEL_IDS_IN_ORDER.map((id) => ({ id, object: "model", owned_by: "claude-code-harness" })) });
			}
			if (req.method === "GET" && path === "/v1/harness/usage") {
				return json(res, 200, await getHarnessUsage());
			}
			if (req.method === "POST" && path === "/v1/chat/completions") {
				return await handleChat(res, await readBody(req, maxBody) as ChatRequest, continuations, streamFn);
			}
			return json(res, 404, { error: { message: "not found" } });
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			if (res.headersSent) {
				if (!res.writableEnded) res.end(`data: ${JSON.stringify({ error: { message } })}\n\ndata: [DONE]\n\n`);
				return;
			}
			return json(res, message.includes("body exceeds") ? 413 : 400, { error: { message } });
		}
	});
	server.listen(port, host, () => console.log(`Claude bridge OpenAI server listening on ${host}:${port}`));
	return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) startServer();
