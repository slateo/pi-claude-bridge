import type {
	AssistantMessage,
	Context,
	ImageContent,
	Message,
	Model,
	TextContent,
	ThinkingLevel,
	Tool,
	ToolCall,
	ToolResultMessage,
	UserMessage,
} from "@earendil-works/pi-ai";
import { captureStandalonePrompt, standaloneModel } from "./index.js";

export type ChatMessage = {
	role?: string;
	content?: unknown;
	tool_call_id?: string;
	name?: string;
	tool_calls?: Array<{
		id?: string;
		type?: string;
		function?: { name?: string; arguments?: string };
	}>;
};

export type ChatTool = {
	type?: string;
	function?: { name?: string; description?: string; parameters?: Record<string, unknown> };
};

export type ChatRequest = {
	model?: string;
	messages?: ChatMessage[];
	tools?: ChatTool[];
	stream?: boolean;
	reasoning_effort?: string;
};

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

function textParts(content: unknown): TextContent[] {
	if (typeof content === "string") return content ? [{ type: "text", text: content }] : [];
	if (!Array.isArray(content)) return [];
	const parts: TextContent[] = [];
	for (const part of content) {
		if (!part || typeof part !== "object") continue;
		const record = part as Record<string, unknown>;
		if ((record.type === "text" || record.type === "input_text" || record.type === "output_text") && typeof record.text === "string") {
			parts.push({ type: "text", text: record.text });
		}
	}
	return parts;
}

function userParts(content: unknown): Array<TextContent | ImageContent> {
	const parts: Array<TextContent | ImageContent> = [...textParts(content)];
	if (!Array.isArray(content)) return parts;
	for (const part of content) {
		if (!part || typeof part !== "object") continue;
		const record = part as Record<string, unknown>;
		if (record.type !== "image_url") continue;
		const imageUrl = record.image_url;
		const url = typeof imageUrl === "string"
			? imageUrl
			: imageUrl && typeof imageUrl === "object" && typeof (imageUrl as Record<string, unknown>).url === "string"
				? String((imageUrl as Record<string, unknown>).url)
				: "";
		const match = /^data:([^;,]+);base64,(.+)$/s.exec(url);
		if (match) parts.push({ type: "image", mimeType: match[1], data: match[2] });
	}
	return parts;
}

function parseArguments(value: unknown): Record<string, unknown> {
	if (value && typeof value === "object") return value as Record<string, unknown>;
	if (typeof value !== "string" || !value.trim()) return {};
	try {
		const parsed = JSON.parse(value);
		return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
	} catch {
		return {};
	}
}

function systemPrompt(messages: ChatMessage[]): string {
	return messages
		.filter((message) => message.role === "system" || message.role === "developer")
		.flatMap((message) => textParts(message.content).map((part) => part.text))
		.join("\n\n");
}

function toolNamesById(messages: ChatMessage[]): Map<string, string> {
	const names = new Map<string, string>();
	for (const message of messages) {
		if (message.role !== "assistant" || !Array.isArray(message.tool_calls)) continue;
		for (const call of message.tool_calls) {
			if (call.id && call.function?.name) names.set(call.id, call.function.name);
		}
	}
	return names;
}

function assistantMessage(message: ChatMessage, modelId: string, timestamp: number): AssistantMessage | undefined {
	const content: Array<TextContent | ToolCall> = [...textParts(message.content)];
	for (const call of message.tool_calls ?? []) {
		if (!call.id || !call.function?.name) continue;
		content.push({
			type: "toolCall",
			id: call.id,
			name: call.function.name,
			arguments: parseArguments(call.function.arguments),
		});
	}
	if (content.length === 0) return undefined;
	return {
		role: "assistant",
		content,
		api: "openai-completions",
		provider: "claude-relay-client",
		model: modelId,
		usage: zeroUsage(),
		stopReason: (message.tool_calls?.length ?? 0) > 0 ? "toolUse" : "stop",
		timestamp,
	};
}

function toolResultMessage(message: ChatMessage, names: Map<string, string>, timestamp: number): ToolResultMessage | undefined {
	if (!message.tool_call_id) return undefined;
	const content = userParts(message.content);
	return {
		role: "toolResult",
		toolCallId: message.tool_call_id,
		toolName: message.name || names.get(message.tool_call_id) || "tool",
		content: content.length > 0 ? content : [{ type: "text", text: "" }],
		isError: false,
		timestamp,
	};
}

function messagesToPi(messages: ChatMessage[], modelId: string): Message[] {
	const names = toolNamesById(messages);
	const result: Message[] = [];
	const base = Date.now() - messages.length;
	for (const [index, message] of messages.entries()) {
		const timestamp = base + index;
		switch (message.role) {
			case "system":
			case "developer":
				break;
			case "assistant": {
				const converted = assistantMessage(message, modelId, timestamp);
				if (converted) result.push(converted);
				break;
			}
			case "tool": {
				const converted = toolResultMessage(message, names, timestamp);
				if (converted) result.push(converted);
				break;
			}
			default: {
				const content = userParts(message.content);
				if (content.length > 0) {
					result.push({ role: "user", content, timestamp } satisfies UserMessage);
				}
			}
		}
	}
	return result;
}

function toolsToPi(tools: ChatTool[] | undefined): Tool[] | undefined {
	if (!tools?.length) return undefined;
	const result: Tool[] = [];
	for (const tool of tools) {
		const fn = tool.function;
		if (tool.type !== "function" || !fn?.name) continue;
		result.push({
			name: fn.name,
			description: fn.description || "",
			parameters: (fn.parameters ?? { type: "object", properties: {} }) as Tool["parameters"],
		});
	}
	return result.length > 0 ? result : undefined;
}

export function reasoningLevel(value: string | undefined): ThinkingLevel | undefined {
	switch (value) {
		case "minimal":
		case "low":
		case "medium":
		case "high":
		case "xhigh":
			return value;
		default:
			return undefined;
	}
}

export function toolResultIds(messages: ChatMessage[]): string[] {
	let toolAssistant = -1;
	let expected = new Set<string>();
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		if (message.tool_calls?.length) {
			toolAssistant = index;
			expected = new Set(message.tool_calls.flatMap((call) => call.id ? [call.id] : []));
		}
		break;
	}
	if (toolAssistant < 0 || expected.size === 0) return [];
	return messages
		.slice(toolAssistant + 1)
		.filter((message) => message.role === "tool" && message.tool_call_id && expected.has(message.tool_call_id))
		.map((message) => message.tool_call_id!);
}

export function chatRequestToRelay(body: ChatRequest, projectedSystemPrompt: string): {
	model: Model<any>;
	context: Context;
	reasoning?: ThinkingLevel;
	resultIds: string[];
} {
	const modelId = String(body.model || "");
	const model = standaloneModel(modelId);
	if (!model) throw new Error(`unsupported model: ${modelId}`);
	if (!Array.isArray(body.messages) || body.messages.length === 0) {
		throw new Error("messages must be a non-empty array");
	}
	if (projectedSystemPrompt) captureStandalonePrompt(projectedSystemPrompt);
	const context: Context = {
		systemPrompt: projectedSystemPrompt || undefined,
		messages: messagesToPi(body.messages, modelId),
		tools: toolsToPi(body.tools),
	};
	if (context.messages.length === 0) throw new Error("messages contain no supported content");
	return { model, context, reasoning: reasoningLevel(body.reasoning_effort), resultIds: toolResultIds(body.messages) };
}

export function rawSystemPrompt(body: ChatRequest): string {
	return systemPrompt(body.messages ?? []);
}

export type ContinuationLease = {
	id: string;
	model: string;
	controller: AbortController;
	expectedIds: Set<string>;
	timer?: NodeJS.Timeout;
};

export class ContinuationRegistry {
	private readonly byToolId = new Map<string, ContinuationLease>();

	constructor(private readonly ttlMs: number) {}

	create(model: string): ContinuationLease {
		return { id: crypto.randomUUID(), model, controller: new AbortController(), expectedIds: new Set() };
	}

	bind(lease: ContinuationLease, ids: string[]): void {
		this.unmap(lease);
		lease.expectedIds = new Set(ids);
		for (const id of lease.expectedIds) this.byToolId.set(id, lease);
		if (lease.timer) clearTimeout(lease.timer);
		lease.timer = setTimeout(() => this.cancel(lease, "Claude relay continuation expired"), this.ttlMs);
		lease.timer.unref();
	}

	resumeOrCreate(model: string, ids: string[]): ContinuationLease {
		if (ids.length === 0) return this.create(model);
		const leases = new Set(ids.map((id) => this.byToolId.get(id)));
		const lease = leases.size === 1 ? [...leases][0] : undefined;
		const resumable = lease
			&& lease.model === model
			&& ids.length === lease.expectedIds.size
			&& ids.every((id) => lease.expectedIds.has(id));
		if (!resumable) {
			for (const stale of leases) if (stale) this.finish(stale);
			return this.create(model);
		}
		this.unmap(lease);
		if (lease.timer) clearTimeout(lease.timer);
		lease.timer = undefined;
		return lease;
	}

	finish(lease: ContinuationLease): void {
		this.unmap(lease);
		if (lease.timer) clearTimeout(lease.timer);
		lease.timer = undefined;
	}

	cancel(lease: ContinuationLease, reason = "Claude relay continuation canceled"): void {
		this.finish(lease);
		lease.controller.abort(new Error(reason));
	}

	get size(): number {
		return new Set(this.byToolId.values()).size;
	}

	private unmap(lease: ContinuationLease): void {
		for (const [id, candidate] of this.byToolId) {
			if (candidate === lease) this.byToolId.delete(id);
		}
	}
}
