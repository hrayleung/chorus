import OpenAI from "openai";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import { LLMMessage, StreamResponseParams } from "../Models";
import { IProvider } from "./IProvider";
import { canProceedWithProvider } from "@core/utilities/ProxyUtils";
import OpenAICompletionsAPIUtils from "@core/chorus/OpenAICompletionsAPIUtils";

function stripThinkBlocks(text: string): string {
    return text
        .replace(/<think(?:\s+[^>]*?)?>[\s\S]*?<\/think\s*>/g, "")
        .replace(/<thought(?:\s+[^>]*?)?>[\s\S]*?<\/thought\s*>/g, "")
        .replace(/<thinkmeta\s+seconds="\d+"\s*\/>/g, "")
        .trim();
}

function stripThinkBlocksFromConversation(
    messages: LLMMessage[],
): LLMMessage[] {
    return messages.map((message) => {
        if (message.role !== "assistant") return message;
        return {
            ...message,
            content: stripThinkBlocks(message.content || ""),
        };
    });
}

export class ProviderFireworks implements IProvider {
    async streamResponse({
        modelConfig,
        llmConversation,
        apiKeys,
        onChunk,
        onComplete,
        additionalHeaders,
        tools,
        customBaseUrl,
    }: StreamResponseParams) {
        const modelName = modelConfig.modelId.split("::")[1];

        const { canProceed, reason } = canProceedWithProvider(
            "fireworks",
            apiKeys,
        );
        if (!canProceed) {
            throw new Error(
                reason || "Please add your Fireworks API key in Settings.",
            );
        }

        const client = new OpenAI({
            baseURL: customBaseUrl || "https://api.fireworks.ai/inference/v1",
            apiKey: apiKeys.fireworks,
            fetch: tauriFetch,
            defaultHeaders: {
                ...(additionalHeaders ?? {}),
            },
            dangerouslyAllowBrowser: true,
        });

        const functionSupport = (tools?.length ?? 0) > 0;
        const sanitizedConversation =
            stripThinkBlocksFromConversation(llmConversation);
        const messages = await OpenAICompletionsAPIUtils.convertConversation(
            sanitizedConversation,
            { imageSupport: true, functionSupport },
        );

        const params: OpenAI.ChatCompletionCreateParamsStreaming & {
            reasoning_effort?: string;
        } = {
            model: modelName,
            messages: [
                ...(modelConfig.systemPrompt
                    ? [
                          {
                              role: "system" as const,
                              content: modelConfig.systemPrompt,
                          },
                      ]
                    : []),
                ...messages,
            ],
            stream: true,
        };

        const normalizedEffort = (
            effort: "low" | "medium" | "high" | "xhigh" | null | undefined,
        ): "low" | "medium" | "high" => {
            if (effort === "low" || effort === "medium" || effort === "high") {
                return effort;
            }
            if (effort === "xhigh") {
                return "high";
            }
            return "medium";
        };

        const lowerModelName = modelName.toLowerCase();
        const canDisableReasoning =
            !lowerModelName.includes("gpt-oss") &&
            !lowerModelName.includes("minimax") &&
            !lowerModelName.includes("m2");

        // Some Fireworks models (notably Kimi) stream full chain-of-thought (and sometimes draft answer text)
        // in `reasoning_content`, which we should not display to users. We'll still show a timed thinking block,
        // but redact the streamed text.
        const redactReasoningContent =
            Boolean(modelConfig.showThoughts) && lowerModelName.includes("kimi");

        if (modelConfig.showThoughts) {
            // Fireworks streams reasoning in `reasoning_content` when enabled.
            // Default to medium when user hasn't set anything.
            params.reasoning_effort = normalizedEffort(
                modelConfig.reasoningEffort,
            );
        } else if (canDisableReasoning) {
            // Best-effort: disable reasoning so we don't pay tokens for it.
            params.reasoning_effort = "none";
        }

        if (tools && tools.length > 0) {
            params.tools =
                OpenAICompletionsAPIUtils.convertToolDefinitions(tools);
            params.tool_choice = "auto";
        }

        const chunks: OpenAI.ChatCompletionChunk[] = [];
        let inReasoning = false;
        let reasoningStartedAtMs: number | undefined;
        let sawNativeReasoning = false;
        let sawContent = false;
        let reasoningBuffer = "";
        let wroteRedactedPlaceholder = false;

        const closeReasoning = () => {
            if (!inReasoning) return;
            inReasoning = false;
            onChunk("</think>");
            if (reasoningStartedAtMs !== undefined) {
                const seconds = Math.max(
                    1,
                    Math.round((Date.now() - reasoningStartedAtMs) / 1000),
                );
                onChunk(`<thinkmeta seconds="${seconds}"/>`);
            }
            reasoningStartedAtMs = undefined;
        };

        const sanitizeTextDelta = (text: string) => {
            if (
                sawNativeReasoning &&
                (text.includes("<think") ||
                    text.includes("</think") ||
                    text.includes("<thought") ||
                    text.includes("</thought"))
            ) {
                return text
                    .replace(/<think/g, "&lt;think")
                    .replace(/<\/think/g, "&lt;/think")
                    .replace(/<thought/g, "&lt;thought")
                    .replace(/<\/thought/g, "&lt;/thought");
            }
            return text;
        };

        let stream: AsyncIterable<OpenAI.ChatCompletionChunk>;
        try {
            stream = await client.chat.completions.create(params);
        } catch (error) {
            // Some Fireworks models reject reasoning_effort="none".
            // If disabling reasoning fails, retry once without the param.
            if (
                !modelConfig.showThoughts &&
                params.reasoning_effort === "none"
            ) {
                delete params.reasoning_effort;
                stream = await client.chat.completions.create(params);
            } else {
                throw error;
            }
        }

        for await (const chunk of stream) {
            chunks.push(chunk);
            const delta = chunk.choices[0]?.delta as unknown as {
                content?: string;
                reasoning?: string;
                reasoning_content?: string;
            };

            const reasoningDelta =
                typeof delta?.reasoning_content === "string"
                    ? delta.reasoning_content
                    : typeof delta?.reasoning === "string"
                      ? delta.reasoning
                      : undefined;

            if (modelConfig.showThoughts && reasoningDelta) {
                sawNativeReasoning = true;
                if (!inReasoning) {
                    inReasoning = true;
                    wroteRedactedPlaceholder = false;
                    reasoningStartedAtMs = Date.now();
                    onChunk("<think>");
                }
                reasoningBuffer += reasoningDelta;
                if (redactReasoningContent) {
                    if (!wroteRedactedPlaceholder) {
                        wroteRedactedPlaceholder = true;
                        onChunk("[redacted]");
                    }
                } else {
                    onChunk(reasoningDelta);
                }
            }

            if (typeof delta?.content === "string" && delta.content) {
                sawContent = true;
                closeReasoning();
                onChunk(sanitizeTextDelta(delta.content));
            }
        }

        closeReasoning();

        // Some models / endpoints may stream all text in a reasoning field (with no `content`).
        // In that case, surface the buffered text as the main output too so users aren't left
        // with an empty assistant message.
        if (modelConfig.showThoughts && !sawContent) {
            const fallbackText = reasoningBuffer.trim();
            if (fallbackText) {
                onChunk("\n\n" + sanitizeTextDelta(fallbackText));
            }
        }

        const toolCalls = OpenAICompletionsAPIUtils.convertToolCalls(
            chunks,
            tools ?? [],
        );

        await onComplete(
            undefined,
            toolCalls.length > 0 ? toolCalls : undefined,
        );
    }
}
