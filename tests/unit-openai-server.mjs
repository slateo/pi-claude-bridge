import assert from "node:assert/strict";
import test from "node:test";
import { parseHarnessUsage, parseUsageResetAt } from "../src/openai-server.ts";

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
