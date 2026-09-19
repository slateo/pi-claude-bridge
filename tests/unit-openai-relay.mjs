import assert from "node:assert/strict";
import test from "node:test";
import { chatRequestToRelay, ContinuationRegistry, toolResultIds } from "../src/openai-relay.ts";

const tool = {
	type: "function",
	function: {
		name: "read",
		description: "Read a file",
		parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
	},
};

test("OpenAI requests retain structured tool calls and results in the Pi context", () => {
	const body = {
		model: "claude-fable-5-1",
		tools: [tool],
		messages: [
			{ role: "system", content: "portable instructions" },
			{ role: "user", content: "read it" },
			{
				role: "assistant",
				content: null,
				tool_calls: [{ id: "toolu_1", type: "function", function: { name: "read", arguments: '{"path":"/tmp/a"}' } }],
			},
			{ role: "tool", tool_call_id: "toolu_1", content: "hello" },
		],
	};
	const relay = chatRequestToRelay(body, "portable instructions");
	assert.deepEqual(relay.resultIds, ["toolu_1"]);
	assert.equal(relay.context.tools[0].name, "read");
	assert.deepEqual(relay.context.messages[1].content[0], {
		type: "toolCall",
		id: "toolu_1",
		name: "read",
		arguments: { path: "/tmp/a" },
	});
	assert.deepEqual(relay.context.messages[2], {
		role: "toolResult",
		toolCallId: "toolu_1",
		toolName: "read",
		content: [{ type: "text", text: "hello" }],
		isError: false,
		timestamp: relay.context.messages[2].timestamp,
	});
});

test("toolResultIds finds all parallel results and ignores completed historical calls", () => {
	assert.deepEqual(toolResultIds([
		{ role: "user", content: "go" },
		{ role: "assistant", tool_calls: [
			{ id: "a", function: { name: "one", arguments: "{}" } },
			{ id: "b", function: { name: "two", arguments: "{}" } },
		] },
		{ role: "tool", tool_call_id: "a", content: "A" },
		{ role: "tool", tool_call_id: "b", content: "B" },
		{ role: "user", content: "steer" },
	]), ["a", "b"]);
	assert.deepEqual(toolResultIds([
		{ role: "assistant", tool_calls: [{ id: "old", function: { name: "one", arguments: "{}" } }] },
		{ role: "tool", tool_call_id: "old", content: "done" },
		{ role: "assistant", content: "finished" },
		{ role: "user", content: "next" },
	]), []);
});

test("an exact continuation match resumes its own lease", () => {
	const registry = new ContinuationRegistry(10_000);
	const lease = registry.create("claude-fable-5-1");
	registry.bind(lease, ["a", "b"]);
	assert.equal(registry.size, 1);
	assert.equal(registry.resumeOrCreate("claude-fable-5-1", ["b", "a"]), lease);
	assert.equal(registry.size, 0);
	registry.finish(lease);
});

test("switching models mid-thread starts a fresh continuation rather than failing the turn", () => {
	const registry = new ContinuationRegistry(10_000);
	const lease = registry.create("claude-fable-5-1");
	registry.bind(lease, ["a", "b"]);
	const switched = registry.resumeOrCreate("claude-opus-5", ["a", "b"]);
	assert.notEqual(switched, lease);
	assert.equal(switched.model, "claude-opus-5");
	assert.equal(switched.controller.signal.aborted, false);
	assert.equal(registry.size, 0);
});

test("unrecognized, partial, and absent continuation ids still yield a usable lease", () => {
	const registry = new ContinuationRegistry(10_000);
	const lease = registry.create("claude-opus-5");
	registry.bind(lease, ["a", "b"]);
	assert.notEqual(registry.resumeOrCreate("claude-opus-5", ["a"]), lease);
	assert.equal(registry.size, 0);
	assert.equal(registry.resumeOrCreate("claude-opus-5", ["toolu_from_another_provider"]).model, "claude-opus-5");
	assert.equal(registry.resumeOrCreate("claude-opus-5", []).model, "claude-opus-5");
	assert.equal(registry.size, 0);
});

test("expired continuation leases abort their Claude query", async () => {
	const registry = new ContinuationRegistry(10);
	const lease = registry.create("claude-fable-5-1");
	registry.bind(lease, ["a"]);
	await new Promise((resolve) => setTimeout(resolve, 30));
	assert.equal(lease.controller.signal.aborted, true);
	assert.equal(registry.size, 0);
});
