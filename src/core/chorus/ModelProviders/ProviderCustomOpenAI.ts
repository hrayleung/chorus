import OpenAI from "openai";
import OpenAICompletionsAPIUtils from "@core/chorus/OpenAICompletionsAPIUtils";
import { SettingsManager } from "@core/utilities/Settings";
import { StreamResponseParams } from "../Models";
import { IProvider, ModelDisabled } from "./IProvider";
import JSON5 from "json5";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";

interface ProviderError {
    message: string;
    error?: {
        message?: string;
        metadata?: { raw?: string };
    };
    metadata?: { raw?: string };
}

function isProviderError(error: unknown): error is ProviderError {
    return (
        typeof error === "object" &&
        error !== null &&
        "message" in error &&
        ("error" in error || "metadata" in error) &&
        error.message === "Provider returned error"
    );
}

function parseCustomProviderModelId(modelId: string): {
    providerId: string;
    modelName: string;
} {
    const parts = modelId.split("::");
    if (parts.length < 3) {
        throw new Error(`Invalid custom provider model id: ${modelId}`);
    }

    const providerId = parts[1] ?? "";
    const modelName = parts.slice(2).join("::");

    if (!providerId || !modelName) {
        throw new Error(`Invalid custom provider model id: ${modelId}`);
    }

    return { providerId, modelName };
}

function getErrorMessage(error: unknown): string {
    if (typeof error === "object" && error !== null && "message" in error) {
        return (error as { message: string }).message;
    } else if (typeof error === "string") {
        return error;
    }
    return "Unknown error";
}

export class ProviderCustomOpenAI implements IProvider {
    async streamResponse({
        llmConversation,
        modelConfig,
        onChunk,
        onComplete,
        additionalHeaders,
        tools,
        onError,
        customBaseUrl,
    }: StreamResponseParams): Promise<ModelDisabled | void> {
        const { providerId, modelName } = parseCustomProviderModelId(
            modelConfig.modelId,
        );

        const settings = await SettingsManager.getInstance().get();
        const provider = (settings.customProviders ?? []).find(
            (p) => p.id === providerId && p.kind === "openai",
        );

        if (!provider) {
            throw new Error(
                "Custom provider not found. Please check your Providers settings.",
            );
        }

        if (!provider.apiBaseUrl.trim()) {
            throw new Error(
                `Please add an API base URL for "${provider.name}" in Settings.`,
            );
        }

        if (!provider.apiKey.trim()) {
            throw new Error(
                `Please add an API key for "${provider.name}" in Settings.`,
            );
        }

        const baseURL = customBaseUrl || provider.apiBaseUrl;

        const client = new OpenAI({
            baseURL,
            apiKey: provider.apiKey,
            fetch: tauriFetch,
            defaultHeaders: {
                ...(additionalHeaders ?? {}),
            },
            dangerouslyAllowBrowser: true,
        });

        let messages: OpenAI.ChatCompletionMessageParam[] =
            await OpenAICompletionsAPIUtils.convertConversation(
                llmConversation,
                {
                    imageSupport:
                        modelConfig.supportedAttachmentTypes?.includes(
                            "image",
                        ) ?? false,
                    functionSupport: true,
                },
            );

        if (modelConfig.systemPrompt) {
            messages = [
                {
                    role: "system",
                    content: modelConfig.systemPrompt,
                },
                ...messages,
            ];
        }

        const streamParams: OpenAI.ChatCompletionCreateParamsStreaming = {
            model: modelName,
            messages,
            stream: true,
        };

        if (tools && tools.length > 0) {
            streamParams.tools =
                OpenAICompletionsAPIUtils.convertToolDefinitions(tools);
            streamParams.tool_choice = "auto";
        }

        const chunks: OpenAI.ChatCompletionChunk[] = [];

        try {
            const stream = await client.chat.completions.create(streamParams);
            for await (const chunk of stream) {
                chunks.push(chunk);
                if (chunk.choices[0]?.delta?.content) {
                    onChunk(chunk.choices[0].delta.content);
                }
            }
        } catch (error: unknown) {
            console.error(
                "Raw error from ProviderCustomOpenAI:",
                error,
                modelName,
                provider.apiBaseUrl,
                messages,
            );
            console.error(JSON.stringify(error, null, 2));

            if (
                isProviderError(error) &&
                error.message === "Provider returned error"
            ) {
                let errorDetails: ProviderError;
                try {
                    errorDetails = JSON5.parse(
                        error.error?.metadata?.raw ||
                            error.metadata?.raw ||
                            "{}",
                    );
                } catch {
                    errorDetails = {
                        message: "Failed to parse error details",
                        error: { message: "Failed to parse error details" },
                    };
                }
                const errorMessage = `Provider returned error: ${errorDetails.error?.message || error.message}`;
                if (onError) {
                    onError(errorMessage);
                } else {
                    throw new Error(errorMessage);
                }
            } else {
                if (onError) {
                    onError(getErrorMessage(error));
                } else {
                    throw error;
                }
            }
            return undefined;
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
