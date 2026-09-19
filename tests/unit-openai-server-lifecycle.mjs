import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { once } from "node:events";
import { request as httpRequest } from "node:http";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import { startServer } from "../src/openai-server.ts";

const waitFor = async (check, timeoutMs = 2_000) => {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (await check()) return;
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	throw new Error("timed out waiting for condition");
};

const envKeys = [
	"CLAUDE_BRIDGE_API_KEY",
	"CLAUDE_BRIDGE_HOST",
	"CLAUDE_BRIDGE_PORT",
	"CLAUDE_BRIDGE_CWD",
	"CLAUDE_BRIDGE_HEARTBEAT_MS",
	"CLAUDE_BRIDGE_TIMEOUT_MS",
];

const applyEnv = (overrides) => {
	const previous = Object.fromEntries(envKeys.map((key) => [key, process.env[key]]));
	Object.assign(process.env, overrides);
	return () => {
		for (const key of envKeys) {
			if (previous[key] === undefined) delete process.env[key];
			else process.env[key] = previous[key];
		}
	};
};

const postCompletion = (port, payload) => new Promise((resolve, reject) => {
	const body = JSON.stringify(payload);
	const request = httpRequest({
		host: "127.0.0.1",
		port,
		path: "/v1/chat/completions",
		method: "POST",
		headers: {
			authorization: "Bearer test-key",
			"content-type": "application/json",
			"content-length": Buffer.byteLength(body),
		},
	}, (response) => {
		let received = "";
		response.setEncoding("utf8");
		response.on("data", (chunk) => { received += chunk; });
		response.on("end", () => resolve({ status: response.statusCode, body: JSON.parse(received) }));
	});
	request.on("error", reject);
	request.end(body);
});

test("streaming requests emit heartbeats and cancel the Claude query when the client disconnects", async () => {
	const directory = await mkdtemp(join(tmpdir(), "openai-bridge-lifecycle-"));
	let aborted = false;
	const streamFn = (_model, _context, options) => ({
		async *[Symbol.asyncIterator]() {
			await new Promise((resolve) => {
				if (options.signal.aborted) return resolve();
				options.signal.addEventListener("abort", resolve, { once: true });
			});
			aborted = true;
		},
	});

	const restoreEnv = applyEnv({
		CLAUDE_BRIDGE_API_KEY: "test-key",
		CLAUDE_BRIDGE_HOST: "127.0.0.1",
		CLAUDE_BRIDGE_PORT: "0",
		CLAUDE_BRIDGE_CWD: directory,
		CLAUDE_BRIDGE_HEARTBEAT_MS: "20",
		CLAUDE_BRIDGE_TIMEOUT_MS: "5000",
	});

	const server = startServer({ streamFn });
	try {
		await once(server, "listening");
		const address = server.address();
		assert.ok(address && typeof address === "object");
		const body = JSON.stringify({
			model: "claude-fable-5-1",
			messages: [{ role: "user", content: "wait" }],
			stream: true,
		});
		let received = "";
		let activeRequest;
		let activeResponse;
		await new Promise((resolve, reject) => {
			const request = httpRequest({
				host: "127.0.0.1",
				port: address.port,
				path: "/v1/chat/completions",
				method: "POST",
				headers: {
					authorization: "Bearer test-key",
					"content-type": "application/json",
					"content-length": Buffer.byteLength(body),
				},
			}, (response) => {
				activeRequest = request;
				activeResponse = response;
				assert.equal(response.statusCode, 200);
				response.setEncoding("utf8");
				response.on("data", (chunk) => {
					received += chunk;
					if ((received.match(/data: /g) ?? []).length >= 2) resolve();
				});
				response.on("error", (error) => {
					if (!response.destroyed) reject(error);
				});
			});
			request.on("error", (error) => {
				if (!request.destroyed) reject(error);
			});
			request.end(body);
		});
		const chunks = received.split("\n\n").filter((chunk) => chunk.startsWith("data: "));
		assert.deepEqual(JSON.parse(chunks[0].slice(6)).choices[0].delta, { role: "assistant" });
		assert.deepEqual(JSON.parse(chunks[1].slice(6)).choices[0].delta, {});
		activeResponse.destroy();
		activeRequest.destroy();
		await waitFor(() => aborted);
	} finally {
		server.closeAllConnections();
		server.close();
		if (server.listening) await once(server, "close");
		restoreEnv();
		await rm(directory, { recursive: true, force: true });
	}
});

test("a tool continuation the bridge never issued starts a fresh turn instead of failing the request", async () => {
	const directory = await mkdtemp(join(tmpdir(), "openai-bridge-continuation-"));
	let turns = 0;
	const streamFn = (_model, context) => {
		turns += 1;
		return {
			async *[Symbol.asyncIterator]() {
				yield {
					type: "done",
					message: {
						role: "assistant",
						content: [{ type: "text", text: `relayed ${context.messages.length} messages` }],
						stopReason: "stop",
					},
				};
			},
		};
	};

	const restoreEnv = applyEnv({
		CLAUDE_BRIDGE_API_KEY: "test-key",
		CLAUDE_BRIDGE_HOST: "127.0.0.1",
		CLAUDE_BRIDGE_PORT: "0",
		CLAUDE_BRIDGE_CWD: directory,
		CLAUDE_BRIDGE_HEARTBEAT_MS: "20",
		CLAUDE_BRIDGE_TIMEOUT_MS: "5000",
	});

	const server = startServer({ streamFn });
	try {
		await once(server, "listening");
		const address = server.address();
		assert.ok(address && typeof address === "object");
		const response = await postCompletion(address.port, {
			model: "claude-opus-5",
			tools: [{
				type: "function",
				function: { name: "read", description: "Read a file", parameters: { type: "object", properties: {} } },
			}],
			messages: [
				{ role: "user", content: "go" },
				{
					role: "assistant",
					content: null,
					tool_calls: [{ id: "call_from_another_provider", type: "function", function: { name: "read", arguments: "{}" } }],
				},
				{ role: "tool", tool_call_id: "call_from_another_provider", content: "contents" },
			],
		});
		assert.equal(response.status, 200);
		assert.equal(response.body.choices[0].message.content, "relayed 3 messages");
		assert.equal(response.body.choices[0].finish_reason, "stop");
		assert.equal(turns, 1);
	} finally {
		server.closeAllConnections();
		server.close();
		if (server.listening) await once(server, "close");
		restoreEnv();
		await rm(directory, { recursive: true, force: true });
	}
});
