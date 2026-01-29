import Together from "together-ai";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import {
    attachmentMissingFlag,
    encodeTextAttachment,
    encodeWebpageAttachment,
    LLMMessage,
    readImageAttachment,
    StreamResponseParams,
} from "../Models";
import { IProvider } from "./IProvider";
import { canProceedWithProvider } from "@core/utilities/ProxyUtils";
import OpenAICompletionsAPIUtils from "@core/chorus/OpenAICompletionsAPIUtils";
import { convertPdfToPng } from "@core/chorus/AttachmentsHelpers";

type TogetherChatMessageParam =
    | Together.Chat.CompletionCreateParams.ChatCompletionSystemMessageParam
    | Together.Chat.CompletionCreateParams.ChatCompletionUserMessageParam
    | Together.Chat.CompletionCreateParams.ChatCompletionAssistantMessageParam
    | Together.Chat.CompletionCreateParams.ChatCompletionToolMessageParam
    | Together.Chat.CompletionCreateParams.ChatCompletionFunctionMessageParam;

function ensureNonEmptyTextParameter(text: string): string {
    return text.trim() === "" ? "..." : text;
}

async function formatMessageWithAttachments(
    message: LLMMessage,
    options: {
        imageSupport: boolean;
        functionSupport: boolean;
    },
): Promise<TogetherChatMessageParam[]> {
    const { imageSupport, functionSupport } = options;

    if (message.role === "tool_results") {
        if (!functionSupport) {
            return [
                {
                    role: "user",
                    content: message.toolResults
                        .map(
                            (result) =>
                                `<tool_result>\n${result.content}\n</tool_result>`,
                        )
                        .join("\n"),
                },
            ];
        }

        return message.toolResults.map((result) => ({
            role: "tool",
            tool_call_id: result.id,
            content: ensureNonEmptyTextParameter(result.content),
        }));
    }

    if (message.role === "assistant") {
        return [
            {
                role: "assistant",
                content: ensureNonEmptyTextParameter(message.content),
                ...(functionSupport && message.toolCalls.length > 0
                    ? {
                          tool_calls: message.toolCalls.map(
                              (toolCall, index) => ({
                                  id: toolCall.id,
                                  index,
                                  type: "function",
                                  function: {
                                      name: toolCall.namespacedToolName,
                                      arguments: JSON.stringify(toolCall.args),
                                  },
                              }),
                          ),
                      }
                    : {}),
            },
        ];
    }

    let attachmentTexts = "";
    const imageContents: Array<
        | Together.Chat.ChatCompletionStructuredMessageText
        | Together.Chat.ChatCompletionStructuredMessageImageURL
        | Together.Chat.ChatCompletionStructuredMessageVideoURL
        | Together.Chat.CompletionCreateParams.ChatCompletionUserMessageParam.Audio
        | Together.Chat.CompletionCreateParams.ChatCompletionUserMessageParam.InputAudio
    > = [];

    for (const attachment of message.attachments) {
        switch (attachment.type) {
            case "text": {
                attachmentTexts += await encodeTextAttachment(attachment);
                break;
            }
            case "webpage": {
                attachmentTexts += await encodeWebpageAttachment(attachment);
                break;
            }
            case "image": {
                if (!imageSupport) {
                    attachmentTexts += attachmentMissingFlag(attachment);
                } else {
                    const fileExt =
                        attachment.path.split(".").pop()?.toLowerCase() || "";
                    const mimeType = fileExt === "jpg" ? "jpeg" : fileExt;
                    imageContents.push({
                        type: "image_url",
                        image_url: {
                            url: `data:image/${mimeType};base64,${await readImageAttachment(attachment)}`,
                        },
                    });
                }
                break;
            }
            case "pdf": {
                try {
                    const pngUrls = await convertPdfToPng(attachment.path);
                    for (const pngUrl of pngUrls) {
                        imageContents.push({
                            type: "image_url",
                            image_url: {
                                url: pngUrl,
                            },
                        });
                    }
                } catch (error) {
                    console.error("Failed to convert PDF to PNG:", error);
                    console.error("PDF path was:", attachment.path);
                }
                break;
            }
            default: {
                // If a provider doesn't support an attachment type, include a text flag so the model can respond anyway.
                attachmentTexts += attachmentMissingFlag(attachment);
            }
        }
    }

    const fullText = ensureNonEmptyTextParameter(
        attachmentTexts + message.content,
    );

    if (imageContents.length > 0) {
        return [
            {
                role: "user",
                content: [{ type: "text", text: fullText }, ...imageContents],
            },
        ];
    }

    return [
        {
            role: "user",
            content: fullText,
        },
    ];
}

export class ProviderTogether implements IProvider {
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
            "together",
            apiKeys,
        );
        if (!canProceed) {
            throw new Error(
                reason || "Please add your Together AI API key in Settings.",
            );
        }

        const client = new Together({
            baseURL: customBaseUrl || "https://api.together.xyz/v1",
            apiKey: apiKeys.together,
            fetch: tauriFetch,
            defaultHeaders: {
                ...(additionalHeaders ?? {}),
            },
        });

        const functionSupport = (tools?.length ?? 0) > 0;
        const imageSupport =
            modelConfig.supportedAttachmentTypes.includes("image");
        const messages: TogetherChatMessageParam[] = [];
        for (const message of llmConversation) {
            messages.push(
                ...(await formatMessageWithAttachments(message, {
                    imageSupport,
                    functionSupport,
                })),
            );
        }

        const systemMessages: Together.Chat.CompletionCreateParams.ChatCompletionSystemMessageParam[] =
            modelConfig.systemPrompt
                ? [{ role: "system", content: modelConfig.systemPrompt }]
                : [];

        const params: Together.Chat.CompletionCreateParamsStreaming = {
            model: modelName,
            messages: [...systemMessages, ...messages],
            stream: true,
        };

        if (tools && tools.length > 0) {
            params.tools =
                OpenAICompletionsAPIUtils.convertToolDefinitions(tools);
            params.tool_choice = "auto";
        }

        const chunks: Together.Chat.ChatCompletionChunk[] = [];
        const stream = await client.chat.completions.create(params);

        for await (const chunk of stream) {
            chunks.push(chunk);
            if (chunk.choices[0]?.delta?.content) {
                onChunk(chunk.choices[0].delta.content);
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
