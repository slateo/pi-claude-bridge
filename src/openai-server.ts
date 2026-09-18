import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { MODEL_IDS_IN_ORDER } from "./models.js";

type ChatMessage = { role?: string; content?: unknown };
type ChatRequest = { model?: string; messages?: ChatMessage[]; stream?: boolean };
type PiRequest = { system: string; prompt: string; sessionJsonl: string; cwd: string };
type PiInvocation = { args: string[]; stdin: string; tempDir: string; cleanup: () => Promise<void> };
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
const DEFAULT_TERMINATION_GRACE_MS = 5_000;
const PI_SYSTEM_PROMPT_START = "You are an expert coding assistant operating inside pi, a coding agent harness.";
const PI_SYSTEM_PROMPT_END = "- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)";
const extensionDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
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

export function messagesToPrompt(messages: ChatMessage[]): { prompt: string; system: string } {
	const system: string[] = [];
	const turns: string[] = [];
	for (const message of messages) {
		const role = typeof message.role === "string" ? message.role : "user";
		const text = textContent(message.content).trim();
		if (!text) continue;
		if (role === "system" || role === "developer") system.push(text);
		else turns.push(`[${role}]\n${text}`);
	}
	return {
		system: system.join("\n\n"),
		prompt: `${turns.join("\n\n")}\n\nRespond to the final user message.`.trim(),
	};
}

function zeroUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Convert prior text turns into a temporary Pi session. The final text turn is
 * delivered over stdin as the live prompt, so neither history nor prompt size
 * is constrained by Linux's per-argument limit.
 */
export function messagesToPiRequest(messages: ChatMessage[], model: string, cwd: string): PiRequest {
	const system: string[] = [];
	const turns: Array<{ role: string; text: string }> = [];
	for (const message of messages) {
		const role = typeof message.role === "string" ? message.role : "user";
		const text = textContent(message.content).trim();
		if (!text) continue;
		if (role === "system" || role === "developer") system.push(text);
		else turns.push({ role, text });
	}
	const current = turns.pop();
	if (!current) return { system: system.join("\n\n"), prompt: "", sessionJsonl: "", cwd };

	const now = Date.now();
	const timestamp = new Date(now).toISOString();
	const sessionId = randomUUID();
	const entries: unknown[] = [{ type: "session", version: 3, id: sessionId, timestamp, cwd }];
	let parentId: string | null = null;
	for (const [index, turn] of turns.entries()) {
		const id = randomUUID().slice(0, 8);
		const messageTimestamp = now - (turns.length - index) * 1000;
		let message: unknown;
		if (turn.role === "assistant") {
			message = {
				role: "assistant",
				content: [{ type: "text", text: turn.text }],
				api: "openai-completions",
				provider: "ai-router",
				model,
				usage: zeroUsage(),
				stopReason: "stop",
				timestamp: messageTimestamp,
			};
		} else {
			message = {
				role: "user",
				content: [{ type: "text", text: turn.role === "user" ? turn.text : `[${turn.role}]\n${turn.text}` }],
				timestamp: messageTimestamp,
			};
		}
		entries.push({ type: "message", id, parentId, timestamp: new Date(messageTimestamp).toISOString(), message });
		parentId = id;
	}

	return {
		system: system.join("\n\n"),
		prompt: current.role === "user" ? current.text : `[${current.role}]\n${current.text}\n\nRespond to this final message.`,
		sessionJsonl: `${entries.map((entry) => JSON.stringify(entry)).join("\n")}\n`,
		cwd,
	};
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

export function parsePiLine(line: string): { delta?: string; thinking?: string; final?: string; usage?: Usage; error?: string } {
	let event: any;
	try { event = JSON.parse(line); } catch { return {}; }
	if (event?.type === "message_update" && event?.assistantMessageEvent?.type === "text_delta") {
		return { delta: event.assistantMessageEvent.delta, usage: event.usage };
	}
	if (event?.type === "message_update" && event?.assistantMessageEvent?.type === "thinking_delta") {
		return { thinking: event.assistantMessageEvent.delta, usage: event.usage };
	}
	if (event?.type === "message_end" && event?.message?.role === "assistant") {
		const final = Array.isArray(event.message.content)
			? event.message.content.filter((part: any) => part?.type === "text").map((part: any) => part.text).join("")
			: "";
		return { final, usage: event.message.usage };
	}
	if (event?.type === "agent_end" && event?.willRetry === false && event?.error) {
		return { error: String(event.error) };
	}
	return {};
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

export async function preparePiInvocation(model: string, request: PiRequest): Promise<PiInvocation> {
	const tempDir = await mkdtemp(join(tmpdir(), "claude-bridge-openai-"));
	await chmod(tempDir, 0o700);
	const sessionPath = join(tempDir, "session.jsonl");
	const systemPath = join(tempDir, "system.txt");
	try {
		await writeFile(sessionPath, request.sessionJsonl, { encoding: "utf8", mode: 0o600 });
		const args = ["--session", sessionPath, "-ne", "-e", process.env.CLAUDE_BRIDGE_EXTENSION_DIR || extensionDir,
			"--model", `claude-bridge/${model}`, "--mode", "json"];
		if (request.system) {
			await writeFile(systemPath, request.system, { encoding: "utf8", mode: 0o600 });
			args.push("--append-system-prompt", systemPath);
		}
		args.push("-p");
		return {
			args,
			stdin: request.prompt,
			tempDir,
			cleanup: () => rm(tempDir, { recursive: true, force: true }),
		};
	} catch (error) {
		await rm(tempDir, { recursive: true, force: true });
		throw error;
	}
}

async function startPi(model: string, request: PiRequest): Promise<{ child: ChildProcessWithoutNullStreams; cleanup: () => Promise<void> }> {
	const piBin = process.env.CLAUDE_BRIDGE_PI_BIN || "pi";
	const invocation = await preparePiInvocation(model, request);
	const child = spawn(piBin, invocation.args, {
		cwd: request.cwd,
		detached: process.platform !== "win32",
		env: { ...process.env, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1" },
		stdio: ["pipe", "pipe", "pipe"],
	});
	child.stdin.on("error", () => {});
	child.stdin.end(invocation.stdin);
	return { child, cleanup: invocation.cleanup };
}

function signalChildTree(child: ChildProcessWithoutNullStreams, signal: NodeJS.Signals): void {
	if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
	if (process.platform !== "win32") {
		try {
			process.kill(-child.pid, signal);
			return;
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ESRCH") return;
		}
	}
	try {
		child.kill(signal);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
	}
}

function terminateChildTree(child: ChildProcessWithoutNullStreams, graceMs: number): NodeJS.Timeout | undefined {
	signalChildTree(child, "SIGTERM");
	if (child.exitCode !== null || child.signalCode !== null) return undefined;
	const forceKill = setTimeout(() => signalChildTree(child, "SIGKILL"), graceMs);
	forceKill.unref();
	return forceKill;
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
	finishReason: "stop" | null = null,
): string {
	return `data: ${JSON.stringify({
		id,
		object: "chat.completion.chunk",
		created,
		model,
		choices: [{ index: 0, delta, finish_reason: finishReason }],
	})}\n\n`;
}

function writeStreamChunk(res: ServerResponse, chunk: string): boolean {
	if (res.destroyed || res.writableEnded) return false;
	res.write(chunk);
	return true;
}

async function handleChat(res: ServerResponse, body: ChatRequest): Promise<void> {
	const model = String(body.model || "");
	if (!modelSet.has(model)) return json(res, 400, { error: { message: `unsupported model: ${model}` } });
	if (!Array.isArray(body.messages) || body.messages.length === 0) {
		return json(res, 400, { error: { message: "messages must be a non-empty array" } });
	}
	const cwd = piClientWorkingDirectory(body.messages, process.env.CLAUDE_BRIDGE_CWD || "/tmp");
	const request = messagesToPiRequest(body.messages, model, cwd);
	request.system = projectPiClientSystemPrompt(request.system);
	if (!request.prompt) return json(res, 400, { error: { message: "messages contain no text" } });

	const id = `chatcmpl-${randomUUID()}`;
	const created = Math.floor(Date.now() / 1000);
	const { child, cleanup } = await startPi(model, request);
	const timeoutMs = Number(process.env.CLAUDE_BRIDGE_TIMEOUT_MS || DEFAULT_TIMEOUT_MS);
	const terminationGraceMs = Number(process.env.CLAUDE_BRIDGE_TERMINATION_GRACE_MS || DEFAULT_TERMINATION_GRACE_MS);
	let terminationReason: string | undefined;
	let forceKill: NodeJS.Timeout | undefined;
	const terminate = (reason: string) => {
		if (terminationReason !== undefined) return;
		terminationReason = reason;
		forceKill = terminateChildTree(child, terminationGraceMs);
	};
	const timeout = setTimeout(() => terminate(`Claude bridge timed out after ${timeoutMs}ms`), timeoutMs);
	timeout.unref();
	let heartbeat: NodeJS.Timeout | undefined;
	let stderr = "";
	let stdoutBuffer = "";
	let final = "";
	let streamed = false;
	let latestUsage: Usage | undefined;

	const onClientClose = () => {
		if (!res.writableEnded) terminate("Claude bridge client disconnected");
	};
	res.once("close", onClientClose);

	if (body.stream) {
		res.writeHead(200, {
			"content-type": "text/event-stream",
			"cache-control": "no-cache, no-transform",
			connection: "keep-alive",
			"x-accel-buffering": "no",
		});
		writeStreamChunk(res, completionChunk(id, created, model, { role: "assistant" }));
		const heartbeatMs = Number(process.env.CLAUDE_BRIDGE_HEARTBEAT_MS || DEFAULT_HEARTBEAT_MS);
		heartbeat = setInterval(() => {
			// Use an empty, valid OpenAI chunk rather than an SSE comment because
			// compatibility proxies may parse and re-encode the stream.
			writeStreamChunk(res, completionChunk(id, created, model, {}));
		}, heartbeatMs);
		heartbeat.unref();
	}

	child.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(-8192); });
	child.stdout.on("data", (chunk) => {
		stdoutBuffer += chunk.toString();
		for (;;) {
			const newline = stdoutBuffer.indexOf("\n");
			if (newline < 0) break;
			const line = stdoutBuffer.slice(0, newline);
			stdoutBuffer = stdoutBuffer.slice(newline + 1);
			const parsed = parsePiLine(line);
			if (parsed.usage) latestUsage = parsed.usage;
			if (parsed.final !== undefined) final = parsed.final;
			if (body.stream && parsed.thinking) {
				writeStreamChunk(res, completionChunk(id, created, model, { reasoning_content: parsed.thinking }));
			}
			if (body.stream && parsed.delta) {
				streamed = true;
				writeStreamChunk(res, completionChunk(id, created, model, { content: parsed.delta }));
			}
		}
	});

	let exitCode: number | null;
	try {
		exitCode = await new Promise<number | null>((resolveExit, reject) => {
			child.once("error", reject);
			child.once("close", resolveExit);
		});
	} finally {
		clearTimeout(timeout);
		if (heartbeat) clearInterval(heartbeat);
		if (forceKill) clearTimeout(forceKill);
		res.off("close", onClientClose);
		await cleanup();
	}
	if (res.destroyed) return;
	if (exitCode !== 0 || !final) {
		const message = terminationReason || stderr.trim().split("\n").slice(-1)[0] || `Claude bridge exited with code ${exitCode}`;
		if (body.stream) {
			writeStreamChunk(res, `data: ${JSON.stringify({ error: { message } })}\n\n`);
			res.end("data: [DONE]\n\n");
			return;
		}
		return json(res, 502, { error: { message } });
	}
	if (body.stream) {
		if (!streamed) writeStreamChunk(res, completionChunk(id, created, model, { content: final }));
		writeStreamChunk(res, completionChunk(id, created, model, {}, "stop"));
		res.end("data: [DONE]\n\n");
		return;
	}
	json(res, 200, {
		id, object: "chat.completion", created, model,
		choices: [{ index: 0, message: { role: "assistant", content: final }, finish_reason: "stop" }],
		usage: usageShape(latestUsage),
	});
}

export function startServer() {
	const apiKey = process.env.CLAUDE_BRIDGE_API_KEY;
	if (!apiKey) throw new Error("CLAUDE_BRIDGE_API_KEY is required");
	const host = process.env.CLAUDE_BRIDGE_HOST || "127.0.0.1";
	const port = Number(process.env.CLAUDE_BRIDGE_PORT || 8318);
	const maxBody = Number(process.env.CLAUDE_BRIDGE_MAX_BODY_BYTES || DEFAULT_MAX_BODY);
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
				return await handleChat(res, await readBody(req, maxBody) as ChatRequest);
			}
			return json(res, 404, { error: { message: "not found" } });
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			return json(res, message.includes("body exceeds") ? 413 : 400, { error: { message } });
		}
	});
	server.listen(port, host, () => console.log(`Claude bridge OpenAI server listening on ${host}:${port}`));
	return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) startServer();
