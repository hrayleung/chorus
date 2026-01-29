import OpenAI from "openai";
import _ from "lodash";
import { StreamResponseParams } from "../Models";
import { IProvider, ModelDisabled } from "./IProvider";
import OpenAICompletionsAPIUtils from "@core/chorus/OpenAICompletionsAPIUtils";
import { canProceedWithProvider } from "@core/utilities/ProxyUtils";
import JSON5 from "json5";
import { fetch } from "@tauri-apps/plugin-http";
import { writeFile, mkdir } from "@tauri-apps/plugin-fs";
import { join, appDataDir } from "@tauri-apps/api/path";
import { convertFileSrc } from "@tauri-apps/api/core";

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

function getGoogleModelName(modelName: string): string {
    if (
        [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.0-flash-lite-preview-02-05",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
        ].includes(modelName)
    ) {
        // allowed model names
        return modelName;
    } else if (modelName === "gemini-2.5-pro-latest") {
        // special case: this is not a real google model name, we just map it to latest thing google has available
        return "gemini-2.5-pro-preview-06-05";
    } else if (modelName === "gemini-2.5-flash-preview-04-17") {
        // alias deprecated preview model to stable version
        return "gemini-2.5-flash";
    }
    // If not found in hardcoded mappings, return the model name as-is
    // (supports dynamically fetched models from API)
    return modelName;
}

// uses OpenAI provider to format the messages
export class ProviderGoogle implements IProvider {
    async streamResponse({
        llmConversation,
        modelConfig,
        onChunk,
        onComplete,
        apiKeys,
        additionalHeaders,
        tools,
        enabledToolsets,
        onError,
        customBaseUrl,
    }: StreamResponseParams): Promise<ModelDisabled | void> {
        const modelName = modelConfig.modelId.split("::")[1];
        const googleModelName = getGoogleModelName(modelName);

        const { canProceed, reason } = canProceedWithProvider(
            "google",
            apiKeys,
        );

        if (!canProceed) {
            throw new Error(
                reason || "Please add your Google AI API key in Settings.",
            );
        }

        const googleApiKey = apiKeys.google;
        if (!googleApiKey) {
            throw new Error("Please add your Google AI API key in Settings.");
        }

        // Check if this is an image generation model
        const isImagenModel = modelName.startsWith("imagen-");
        const isGeminiImageModel = modelName.includes("-image-preview");

        if (isImagenModel) {
            // Use Imagen API for text-to-image generation
            await generateImageWithImagen({
                modelName,
                llmConversation,
                apiKey: googleApiKey,
                onChunk,
                onComplete,
                onError,
            });
            return;
        }

        if (isGeminiImageModel) {
            // Use Gemini with image generation capabilities
            await generateImageWithGemini({
                modelName,
                llmConversation,
                apiKey: googleApiKey,
                onChunk,
                onComplete,
                onError,
            });
            return;
        }

        const nativeWebSearchEnabled =
            enabledToolsets?.includes("web") ?? false;

        if (nativeWebSearchEnabled) {
            await streamGroundedResponseWithGoogleSearch({
                googleModelName,
                llmConversation,
                systemPrompt: modelConfig.systemPrompt,
                apiKey: googleApiKey,
                onChunk,
                onComplete,
                onError,
                customBaseUrl,
            });
            return;
        }

        // Google AI uses the generativelanguage.googleapis.com endpoint with OpenAI compatibility
        const baseURL =
            customBaseUrl ||
            "https://generativelanguage.googleapis.com/v1beta/openai";

        // unset headers that are not supported by the Google API
        // https://discuss.ai.google.dev/t/gemini-api-cors-error-with-openai-compatability/58619/16
        const headers = {
            ...(additionalHeaders ?? {}),
            "x-stainless-arch": null,
            "x-stainless-lang": null,
            "x-stainless-os": null,
            "x-stainless-package-version": null,
            "x-stainless-retry-count": null,
            "x-stainless-runtime": null,
            "x-stainless-runtime-version": null,
            "x-stainless-timeout": null,
        };
        const client = new OpenAI({
            baseURL,
            apiKey: googleApiKey,
            defaultHeaders: headers,
            dangerouslyAllowBrowser: true,
        });

        let messages: OpenAI.ChatCompletionMessageParam[] =
            await OpenAICompletionsAPIUtils.convertConversation(
                llmConversation,
                {
                    imageSupport: true,
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
            model: googleModelName,
            messages: messages,
            stream: true,
        };

        // Add Gemini thinking parameters
        const isGemini3 = googleModelName.includes("gemini-3");
        const isGemini25 = googleModelName.includes("gemini-2.5");

        if (isGemini3 && modelConfig.thinkingLevel) {
            // Gemini 3 uses thinking_level parameter
            (
                streamParams as unknown as Record<string, unknown>
            ).thinking_level = modelConfig.thinkingLevel;
        } else if (isGemini25 && modelConfig.budgetTokens) {
            // Gemini 2.5 uses thinking_budget parameter
            (
                streamParams as unknown as Record<string, unknown>
            ).thinking_budget = modelConfig.budgetTokens;
        }

        // Debug: Log thinking parameters
        console.log(`[ProviderGoogle] Model: ${googleModelName}`);
        console.log(
            `[ProviderGoogle] isGemini3: ${isGemini3}, isGemini25: ${isGemini25}`,
        );
        console.log(
            `[ProviderGoogle] modelConfig.thinkingLevel: ${modelConfig.thinkingLevel}`,
        );
        console.log(
            `[ProviderGoogle] modelConfig.budgetTokens: ${modelConfig.budgetTokens}`,
        );
        console.log(`[ProviderGoogle] streamParams thinking params:`, {
            thinking_level: (streamParams as unknown as Record<string, unknown>)
                .thinking_level,
            thinking_budget: (
                streamParams as unknown as Record<string, unknown>
            ).thinking_budget,
        });

        // Add tools definitions
        if (tools && tools.length > 0) {
            streamParams.tools =
                OpenAICompletionsAPIUtils.convertToolDefinitions(tools);
            streamParams.tool_choice = "auto";
        }

        const chunks = [];

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
                "Raw error from ProviderGoogle:",
                error,
                modelName,
                messages,
            );
            console.error(JSON.stringify(error, null, 2));

            if (
                isProviderError(error) &&
                error.message === "Provider returned error"
            ) {
                const errorDetails: ProviderError = JSON5.parse(
                    error.error?.metadata?.raw || error.metadata?.raw || "{}",
                );
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

function getErrorMessage(error: unknown): string {
    if (typeof error === "object" && error !== null && "message" in error) {
        return (error as { message: string }).message;
    } else if (typeof error === "string") {
        return error;
    } else {
        return "Unknown error";
    }
}

async function streamGroundedResponseWithGoogleSearch(params: {
    googleModelName: string;
    llmConversation: StreamResponseParams["llmConversation"];
    systemPrompt: string | undefined;
    apiKey: string;
    onChunk: StreamResponseParams["onChunk"];
    onComplete: StreamResponseParams["onComplete"];
    onError: StreamResponseParams["onError"];
    customBaseUrl: string | undefined;
}): Promise<void> {
    try {
        const { googleModelName, llmConversation, systemPrompt, apiKey } =
            params;

        const messages = await OpenAICompletionsAPIUtils.convertConversation(
            llmConversation,
            {
                imageSupport: false,
                functionSupport: false,
            },
        );

        const contents: Array<{
            role: "user" | "model";
            parts: Array<{ text: string }>;
        }> = messages.map((m) => {
            const text = normalizeToText(m.content);
            return {
                role: m.role === "assistant" ? "model" : "user",
                parts: [{ text: text.trim() === "" ? "..." : text }],
            };
        });

        if (systemPrompt) {
            if (contents.length > 0 && contents[0].role === "user") {
                contents[0].parts[0].text =
                    `${systemPrompt}\n\n` + contents[0].parts[0].text;
            } else {
                contents.unshift({
                    role: "user",
                    parts: [{ text: systemPrompt }],
                });
            }
        }

        const geminiBaseUrl = params.customBaseUrl
            ? params.customBaseUrl.replace(/\/openai\/?$/, "")
            : "https://generativelanguage.googleapis.com/v1beta";

        const response = await fetch(
            `${geminiBaseUrl}/models/${googleModelName}:generateContent`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "x-goog-api-key": apiKey,
                },
                body: JSON.stringify({
                    contents,
                    tools: [
                        {
                            google_search: {},
                        },
                    ],
                }),
            },
        );

        if (!response.ok) {
            const errorBody = await safeJson(response);
            throw new Error(
                getGeminiErrorMessageFromBody(errorBody) ?? response.statusText,
            );
        }

        const data = (await response.json()) as unknown;
        const { text, sources } = extractGeminiTextAndSources(data);

        if (text) {
            params.onChunk(text);
        }
        if (sources.length > 0) {
            const sourcesText = sources
                .map((s, i) => `${i + 1}. [${s.title || s.url}](${s.url})`)
                .join("\n");
            params.onChunk(`\n\nSources:\n${sourcesText}`);
        }

        await params.onComplete();
    } catch (error) {
        console.error("Error using Gemini google_search grounding:", error);
        params.onError(getErrorMessage(error));
    }
}

function normalizeToText(
    content: OpenAI.ChatCompletionMessageParam["content"],
): string {
    if (typeof content === "string") {
        return content;
    }

    if (Array.isArray(content)) {
        return content
            .map((part) => (isOpenAITextPart(part) ? part.text : ""))
            .filter(Boolean)
            .join("\n");
    }

    try {
        return JSON.stringify(content);
    } catch {
        return String(content);
    }
}

async function safeJson(response: Response): Promise<unknown> {
    try {
        return (await response.json()) as unknown;
    } catch {
        return undefined;
    }
}

function getGeminiErrorMessageFromBody(body: unknown): string | undefined {
    if (!isPlainObject(body)) {
        return undefined;
    }

    const errorValue = body["error"];
    if (!isPlainObject(errorValue)) {
        return undefined;
    }

    const messageValue = errorValue["message"];
    return typeof messageValue === "string" ? messageValue : undefined;
}

function extractGeminiTextAndSources(data: unknown): {
    text: string;
    sources: Array<{ url: string; title?: string }>;
} {
    if (!isPlainObject(data)) {
        return { text: "", sources: [] };
    }

    const candidatesValue = data["candidates"];
    if (!isUnknownArray(candidatesValue) || candidatesValue.length === 0) {
        return { text: "", sources: [] };
    }

    const firstCandidate = candidatesValue[0];
    if (!isPlainObject(firstCandidate)) {
        return { text: "", sources: [] };
    }

    const text = extractTextFromCandidate(firstCandidate);
    const grounding = firstCandidate["groundingMetadata"];
    const sources: Array<{ url: string; title?: string }> = [];

    if (isPlainObject(grounding)) {
        const groundingChunksValue = grounding["groundingChunks"];
        if (Array.isArray(groundingChunksValue)) {
            for (const chunk of groundingChunksValue) {
                if (!isPlainObject(chunk)) continue;
                const webValue = chunk["web"];
                if (!isPlainObject(webValue)) continue;
                const uriValue = webValue["uri"];
                if (typeof uriValue !== "string" || !uriValue) continue;
                const titleValue = webValue["title"];
                sources.push({
                    url: uriValue,
                    title:
                        typeof titleValue === "string" ? titleValue : undefined,
                });
            }
        }

        const webResultsValue = grounding["webResults"];
        if (Array.isArray(webResultsValue)) {
            for (const result of webResultsValue) {
                if (!isPlainObject(result)) continue;
                const urlValue = result["url"];
                if (typeof urlValue !== "string" || !urlValue) continue;
                const titleValue = result["title"];
                sources.push({
                    url: urlValue,
                    title:
                        typeof titleValue === "string" ? titleValue : undefined,
                });
            }
        }

        const citationsValue = grounding["citations"];
        if (Array.isArray(citationsValue)) {
            for (const citation of citationsValue) {
                if (!isPlainObject(citation)) continue;
                const uriValue = citation["uri"];
                if (typeof uriValue !== "string" || !uriValue) continue;
                sources.push({ url: uriValue });
            }
        }
    }

    const deduped = new Map<string, { url: string; title?: string }>();
    for (const source of sources) {
        if (!deduped.has(source.url)) {
            deduped.set(source.url, source);
        }
    }

    return {
        text,
        sources: Array.from(deduped.values()),
    };
}

function extractTextFromCandidate(candidate: Record<string, unknown>): string {
    const contentValue = candidate["content"];
    if (!isPlainObject(contentValue)) {
        return "";
    }

    const partsValue = contentValue["parts"];
    if (!isUnknownArray(partsValue)) {
        return "";
    }

    return partsValue
        .map((part) => {
            if (!isPlainObject(part)) return "";
            const textValue = part["text"];
            return typeof textValue === "string" ? textValue : "";
        })
        .filter(Boolean)
        .join("")
        .trim();
}

type OpenAITextPart = { type: "text"; text: string };

function isOpenAITextPart(part: unknown): part is OpenAITextPart {
    return (
        isPlainObject(part) &&
        part["type"] === "text" &&
        typeof part["text"] === "string"
    );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUnknownArray(value: unknown): value is unknown[] {
    return Array.isArray(value);
}

/**
 * Generate image using Imagen API (imagen-4.0 models)
 */
async function generateImageWithImagen(params: {
    modelName: string;
    llmConversation: StreamResponseParams["llmConversation"];
    apiKey: string;
    onChunk: StreamResponseParams["onChunk"];
    onComplete: StreamResponseParams["onComplete"];
    onError: StreamResponseParams["onError"];
}): Promise<void> {
    try {
        // Get the last user message as the prompt
        const lastMessage = params.llmConversation[params.llmConversation.length - 1];
        if (!lastMessage || lastMessage.role !== "user") {
            throw new Error("No user prompt found for image generation");
        }

        const prompt = lastMessage.content;

        // Call Imagen API
        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/${params.modelName}:predict`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "x-goog-api-key": params.apiKey,
                },
                body: JSON.stringify({
                    instances: [{ prompt }],
                    parameters: {
                        sampleCount: 1,
                    },
                }),
            },
        );

        if (!response.ok) {
            const errorData = await response.text();
            throw new Error(`Imagen API error: ${errorData}`);
        }

        const data = (await response.json()) as {
            predictions?: Array<{ bytesBase64Encoded?: string }>;
        };

        if (!data.predictions || data.predictions.length === 0) {
            throw new Error("No image generated");
        }

        const imageData = data.predictions[0]?.bytesBase64Encoded;
        if (!imageData) {
            throw new Error("No image data in response");
        }

        // Save image to disk
        const imagePath = await saveBase64Image(imageData, prompt);

        // Return markdown with image
        const imageUrl = convertFileSrc(imagePath);
        params.onChunk(`![Generated Image](${imageUrl})\n\nImage saved to: ${imagePath}`);
        await params.onComplete();
    } catch (error) {
        console.error("Error generating image with Imagen:", error);
        params.onError(getErrorMessage(error));
    }
}

/**
 * Generate image using Gemini with image generation capabilities
 */
async function generateImageWithGemini(params: {
    modelName: string;
    llmConversation: StreamResponseParams["llmConversation"];
    apiKey: string;
    onChunk: StreamResponseParams["onChunk"];
    onComplete: StreamResponseParams["onComplete"];
    onError: StreamResponseParams["onError"];
}): Promise<void> {
    try {
        // Get the last user message as the prompt
        const lastMessage = params.llmConversation[params.llmConversation.length - 1];
        if (!lastMessage || lastMessage.role !== "user") {
            throw new Error("No user prompt found for image generation");
        }

        const prompt = lastMessage.content;

        // Build contents array for Gemini API
        const contents = [
            {
                role: "user",
                parts: [{ text: prompt }],
            },
        ];

        // Call Gemini API with response_modalities: ["IMAGE"]
        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/${params.modelName}:generateContent`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "x-goog-api-key": params.apiKey,
                },
                body: JSON.stringify({
                    contents,
                    generationConfig: {
                        responseModalities: ["IMAGE"],
                    },
                }),
            },
        );

        if (!response.ok) {
            const errorData = await response.text();
            throw new Error(`Gemini API error: ${errorData}`);
        }

        const data = (await response.json()) as {
            candidates?: Array<{
                content?: {
                    parts?: Array<{
                        inlineData?: {
                            data?: string;
                            mimeType?: string;
                        };
                    }>;
                };
            }>;
        };

        // Extract image from response
        const candidate = data.candidates?.[0];
        const parts = candidate?.content?.parts;
        const imagePart = parts?.find((part) => part.inlineData?.data);

        if (!imagePart?.inlineData?.data) {
            throw new Error("No image generated");
        }

        const imageData = imagePart.inlineData.data;

        // Save image to disk
        const imagePath = await saveBase64Image(imageData, prompt);

        // Return markdown with image
        const imageUrl = convertFileSrc(imagePath);
        params.onChunk(`![Generated Image](${imageUrl})\n\nImage saved to: ${imagePath}`);
        await params.onComplete();
    } catch (error) {
        console.error("Error generating image with Gemini:", error);
        params.onError(getErrorMessage(error));
    }
}

/**
 * Save base64 image data to disk and return file path
 */
async function saveBase64Image(
    base64Data: string,
    prompt: string,
): Promise<string> {
    // Decode base64 to bytes
    const byteString = atob(base64Data);
    const byteArray = new Uint8Array(byteString.length);
    for (let i = 0; i < byteString.length; i++) {
        byteArray[i] = byteString.charCodeAt(i);
    }

    // Create directory for generated images
    const appCoreDir = await appDataDir();
    const imagesDir = await join(appCoreDir, "generated_images");
    await mkdir(imagesDir, { recursive: true });

    // Generate filename
    const timestamp = Date.now().toString().slice(-5);
    const slugifiedPrompt = prompt
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 100);
    const fileName = `${slugifiedPrompt}-${timestamp}.png`;
    const filePath = await join(imagesDir, fileName);

    // Write file
    await writeFile(filePath, byteArray);

    return filePath;
}
