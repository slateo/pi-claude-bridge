import assert from "node:assert/strict";
import test from "node:test";
import { classifyTerminalError, parseHarnessUsage, parseUsageResetAt } from "../src/openai-server.ts";

test("parseUsageResetAt respects the timezone in Claude Code output", () => {
	assert.equal(
		parseUsageResetAt("Sep 16, 8pm (Asia/Tokyo)", new Date("2026-09-16T07:00:00Z")),
		"2026-09-16T11:00:00.000Z",
	);
});

test("parseHarnessUsage maps Claude Code quota text to Anthropic quota windows", () => {
	const usage = parseHarnessUsage([
		"Current session: 5% used · resets Sep 16, 8pm (Asia/Tokyo)",
		"Current week (all models): 11% used · resets Sep 22, 12am (Asia/Tokyo)",
		"Current week (Fable): 3% used · resets Sep 22, 12am (Asia/Tokyo)",
	].join("\n"), new Date("2026-09-16T07:00:00Z"));
	assert.deepEqual(usage, {
		five_hour: { utilization: 5, resets_at: "2026-09-16T11:00:00.000Z" },
		seven_day: { utilization: 11, resets_at: "2026-09-21T15:00:00.000Z" },
		limits: [{
			kind: "weekly_scoped",
			percent: 3,
			resets_at: "2026-09-21T15:00:00.000Z",
			is_active: true,
			scope: { model: { display_name: "Fable" } },
		}],
	});
});

test("classifyTerminalError maps context overflow and payload limits to 4xx instead of 502", () => {
	assert.deepEqual(classifyTerminalError("Prompt is too long"), {
		status: 400,
		type: "invalid_request_error",
		code: "context_length_exceeded",
	});
	assert.deepEqual(classifyTerminalError("prompt is too long: 1020196 tokens > 1000000 maximum"), {
		status: 400,
		type: "invalid_request_error",
		code: "context_length_exceeded",
	});
	assert.deepEqual(classifyTerminalError("request body exceeds limit"), {
		status: 413,
		type: "invalid_request_error",
		code: "request_too_large",
	});
	assert.deepEqual(classifyTerminalError("You're out of extra usage"), {
		status: 429,
		type: "rate_limit_error",
	});
	assert.deepEqual(classifyTerminalError("Claude relay ended without a final message"), {
		status: 502,
		type: "server_error",
	});
});
