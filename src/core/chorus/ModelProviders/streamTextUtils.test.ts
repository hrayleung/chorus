import { describe, expect, it } from "vitest";

import { isMeaningfulTextDelta } from "./streamTextUtils";

describe("isMeaningfulTextDelta", () => {
    it("returns false for whitespace-only chunks", () => {
        expect(isMeaningfulTextDelta("")).toBe(false);
        expect(isMeaningfulTextDelta(" ")).toBe(false);
        expect(isMeaningfulTextDelta("\n")).toBe(false);
        expect(isMeaningfulTextDelta("\t")).toBe(false);
        expect(isMeaningfulTextDelta(" \n\t ")).toBe(false);
    });

    it("returns true when a chunk contains non-whitespace", () => {
        expect(isMeaningfulTextDelta("a")).toBe(true);
        expect(isMeaningfulTextDelta(" a ")).toBe(true);
        expect(isMeaningfulTextDelta("\nfoo")).toBe(true);
    });
});

