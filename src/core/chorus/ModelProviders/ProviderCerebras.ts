import OpenAI from "openai";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import { StreamResponseParams } from "../Models";
import { IProvider } from "./IProvider";
import { canProceedWithProvider } from "@core/utilities/ProxyUtils";
import OpenAICompletionsAPIUtils from "@core/chorus/OpenAICompletionsAPIUtils";
import * as Prompts from "@core/chorus/prompts/prompts";

export class ProviderCerebras implements IProvider {
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
            "cerebras",
            apiKeys,
        );
        if (!canProceed) {
            throw new Error(
                reason || "Please add your Cerebras API key in Settings.",
            );
        }

        const client = new OpenAI({
            baseURL: customBaseUrl || "https://api.cerebras.ai/v1",
            apiKey: apiKeys.cerebras,
            fetch: tauriFetch,
            defaultHeaders: {
                ...(additionalHeaders ?? {}),
            },
            dangerouslyAllowBrowser: true,
        });

        const functionSupport = (tools?.length ?? 0) > 0;
        const messages = await OpenAICompletionsAPIUtils.convertConversation(
            llmConversation,
            {
                imageSupport: false,
                functionSupport,
            },
        );

        const lowerModelName = modelName.toLowerCase();
        const isZaiGlm = lowerModelName.includes("glm");
        const isGptOss = lowerModelName.includes("gpt-oss");
        const isQwen3 = lowerModelName.includes("qwen3");
        const isReasoningModel = isZaiGlm || isGptOss || isQwen3;
        const canSafelyEnableNativeReasoning =
            isReasoningModel && !functionSupport; // Cerebras docs: tool calling + reasoning streaming may be unsupported.

        // Auto-detect if thinking should be shown:
        // - For GLM: use showThoughts (explicit enable/disable toggle)
        // - For GPT-OSS/Qwen3: auto-enable if reasoningEffort is set
        const shouldShowThoughts =
            (isZaiGlm && modelConfig.showThoughts) ||
            (isGptOss && modelConfig.reasoningEffort) ||
            (isQwen3 && modelConfig.reasoningEffort);

        const systemPrompt = [
            shouldShowThoughts && !canSafelyEnableNativeReasoning
                ? Prompts.THOUGHTS_SYSTEM_PROMPT
                : undefined,
            modelConfig.systemPrompt,
        ]
            .filter(Boolean)
            .join("\n\n");

        const params: OpenAI.ChatCompletionCreateParamsStreaming & {
            disable_reasoning?: boolean;
            clear_thinking?: boolean;
            reasoning_format?: "parsed" | "raw" | "hidden" | "none";
            reasoning_effort?: "low" | "medium" | "high";
        } = {
            model: modelName,
            messages: [
                ...(systemPrompt
                    ? [
                          {
                              role: "system" as const,
                              content: systemPrompt,
                          },
                      ]
                    : []),
                ...messages,
            ],
            stream: true,
        };

        // Apply reasoning parameters based on model type
        if (isReasoningModel && canSafelyEnableNativeReasoning) {
            // GPT-OSS: Don't set reasoning_format, use model default (text_parsed)
            // This lets the model naturally separate thinking from answer
            if (!isGptOss) {
                if (shouldShowThoughts) {
                    // GLM/Qwen3: Use 'parsed' format (separate reasoning field)
                    params.reasoning_format = "parsed";
                } else {
                    // Hide reasoning to save tokens when not needed
                    params.reasoning_format = "hidden";
                }
            } else if (!shouldShowThoughts) {
                // GPT-OSS: Only set hidden format when thoughts are disabled
                params.reasoning_format = "hidden";
            }
        }

        // GPT-OSS specific: reasoning_effort parameter
        if (isGptOss && modelConfig.reasoningEffort) {
            const effort = modelConfig.reasoningEffort;
            // Map 'xhigh' to 'high' since Cerebras only supports low/medium/high
            params.reasoning_effort = effort === "xhigh" ? "high" : effort;
        }

        // GLM specific: legacy disable_reasoning and clear_thinking parameters
        if (isZaiGlm) {
            // Cerebras GLM supports non-standard flags (OpenAI-compatible endpoints differ here).
            // When showThoughts is off, disable reasoning to avoid token spend.
            // When showThoughts is on, explicitly enable reasoning (when safe).
            if (!modelConfig.showThoughts || canSafelyEnableNativeReasoning) {
                params.disable_reasoning = !modelConfig.showThoughts;
                params.clear_thinking = !modelConfig.showThoughts;
            }
        }

        if (tools && tools.length > 0) {
            params.tools =
                OpenAICompletionsAPIUtils.convertToolDefinitions(tools);
            params.tool_choice = "auto";
        }

        const chunks: OpenAI.ChatCompletionChunk[] = [];
        let inReasoning = false;
        let reasoningStartedAtMs: number | undefined;
        let reasoningHasNativeTags = false; // Track if reasoning field already contains <think> tags

        const closeReasoning = () => {
            if (!inReasoning) return;
            inReasoning = false;
            if (!reasoningHasNativeTags) {
                // Only add closing tag if we added the opening tag
                onChunk("</think>");
                if (reasoningStartedAtMs !== undefined) {
                    const seconds = Math.max(
                        1,
                        Math.round((Date.now() - reasoningStartedAtMs) / 1000),
                    );
                    onChunk(`<thinkmeta seconds="${seconds}"/>`);
                }
            }
            reasoningStartedAtMs = undefined;
            reasoningHasNativeTags = false;
        };
        let stream: AsyncIterable<OpenAI.ChatCompletionChunk>;
        try {
            stream = await client.chat.completions.create(params);
        } catch (error) {
            // Some OpenAI-compatible endpoints reject non-standard flags.
            // If that happens, retry once without them.
            const hasReasoningParams =
                params.disable_reasoning !== undefined ||
                params.clear_thinking !== undefined ||
                params.reasoning_format !== undefined ||
                params.reasoning_effort !== undefined;

            if (hasReasoningParams) {
                delete params.disable_reasoning;
                delete params.clear_thinking;
                delete params.reasoning_format;
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

            if (shouldShowThoughts && reasoningDelta) {
                if (!inReasoning) {
                    inReasoning = true;
                    reasoningStartedAtMs = Date.now();
                    // Check if reasoning already contains native <think> tags
                    const hasNativeTags =
                        reasoningDelta.includes("<think") ||
                        reasoningDelta.includes("</think") ||
                        reasoningDelta.includes("<thought");
                    reasoningHasNativeTags = hasNativeTags;

                    // Only add our <think> tag if model didn't provide one
                    if (!hasNativeTags) {
                        onChunk("<think>");
                    }
                }
                onChunk(reasoningDelta);
            }

            if (typeof delta?.content === "string" && delta.content) {
                closeReasoning();
                onChunk(delta.content);
            }
        }

        closeReasoning();

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
