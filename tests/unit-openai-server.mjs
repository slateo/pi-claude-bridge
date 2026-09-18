import assert from "node:assert/strict";
import test from "node:test";
import { messagesToPrompt, parseHarnessUsage, parsePiLine, parseUsageResetAt } from "../src/openai-server.ts";

test("messagesToPrompt separates system messages and preserves turns", () => {
	const result = messagesToPrompt([
		{ role: "system", content: "Be terse." },
		{ role: "user", content: "Hello" },
		{ role: "assistant", content: "Hi" },
		{ role: "user", content: [{ type: "text", text: "Final" }] },
	]);
	assert.equal(result.system, "Be terse.");
	assert.match(result.prompt, /\[user\]\nHello/);
	assert.match(result.prompt, /\[assistant\]\nHi/);
	assert.match(result.prompt, /\[user\]\nFinal/);
});

test("parsePiLine extracts text deltas and final text", () => {
	assert.deepEqual(parsePiLine(JSON.stringify({
		type: "message_update",
		usage: { input: 1, output: 2 },
		assistantMessageEvent: { type: "text_delta", delta: "hello" },
	})), { delta: "hello", usage: { input: 1, output: 2 } });
	assert.deepEqual(parsePiLine(JSON.stringify({
		type: "message_update",
		usage: { input: 2, output: 3 },
		assistantMessageEvent: { type: "thinking_delta", delta: "checking" },
	})), { thinking: "checking", usage: { input: 2, output: 3 } });
	assert.deepEqual(parsePiLine(JSON.stringify({
		type: "message_end",
		message: { role: "assistant", usage: { output: 3 }, content: [{ type: "thinking", thinking: "hidden" }, { type: "text", text: "done" }] },
	})), { final: "done", usage: { output: 3 } });
});

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
