import { Toolset as Toolset } from "@core/chorus/Toolsets";

export class ToolsetWeb extends Toolset {
    constructor() {
        super(
            "web",
            "Web",
            {},
            "Enable native web search (OpenAI, Vertex, Gemini, Claude, Grok).",
            "",
        );
    }
}
