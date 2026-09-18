import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

test("streaming requests emit heartbeats and reap Pi when the client disconnects", async () => {
	const directory = await mkdtemp(join(tmpdir(), "openai-bridge-lifecycle-"));
	const fakePi = join(directory, "fake-pi.mjs");
	const ready = join(directory, "ready");
	const terminated = join(directory, "terminated");
	await writeFile(fakePi, `#!/usr/bin/env node
import { writeFileSync } from "node:fs";
process.on("SIGTERM", () => {
  writeFileSync(process.env.FAKE_PI_TERMINATED, "SIGTERM");
  process.exit(0);
});
writeFileSync(process.env.FAKE_PI_READY, "ready");
process.stdin.resume();
setInterval(() => {}, 1000);
`);
	await chmod(fakePi, 0o700);

	const keys = [
		"CLAUDE_BRIDGE_API_KEY",
		"CLAUDE_BRIDGE_HOST",
		"CLAUDE_BRIDGE_PORT",
		"CLAUDE_BRIDGE_PI_BIN",
		"CLAUDE_BRIDGE_CWD",
		"CLAUDE_BRIDGE_HEARTBEAT_MS",
		"CLAUDE_BRIDGE_TIMEOUT_MS",
		"CLAUDE_BRIDGE_TERMINATION_GRACE_MS",
		"FAKE_PI_READY",
		"FAKE_PI_TERMINATED",
	];
	const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
	Object.assign(process.env, {
		CLAUDE_BRIDGE_API_KEY: "test-key",
		CLAUDE_BRIDGE_HOST: "127.0.0.1",
		CLAUDE_BRIDGE_PORT: "0",
		CLAUDE_BRIDGE_PI_BIN: fakePi,
		CLAUDE_BRIDGE_CWD: directory,
		CLAUDE_BRIDGE_HEARTBEAT_MS: "20",
		CLAUDE_BRIDGE_TIMEOUT_MS: "5000",
		CLAUDE_BRIDGE_TERMINATION_GRACE_MS: "100",
		FAKE_PI_READY: ready,
		FAKE_PI_TERMINATED: terminated,
	});

	const server = startServer();
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
		await waitFor(async () => {
			try {
				return (await readFile(ready, "utf8")) === "ready";
			} catch {
				return false;
			}
		});
		activeResponse.destroy();
		activeRequest.destroy();

		await waitFor(async () => {
			try {
				return (await readFile(terminated, "utf8")) === "SIGTERM";
			} catch {
				return false;
			}
		});
	} finally {
		server.closeAllConnections();
		server.close();
		if (server.listening) await once(server, "close");
		for (const key of keys) {
			if (previous[key] === undefined) delete process.env[key];
			else process.env[key] = previous[key];
		}
		await rm(directory, { recursive: true, force: true });
	}
});
