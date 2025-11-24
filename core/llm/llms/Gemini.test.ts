import { describe, expect, it } from "vitest";
import Gemini from "./Gemini.js";
import { ChatMessage, CompletionOptions } from "../../index.js";

describe("Gemini thought signature support", () => {
  const mockGemini = new Gemini({
    apiKey: "test-key",
    model: "gemini-2.5-pro",
  });

  describe("prepareBody", () => {
    it("should attach thought signature to first tool call when present", () => {
      const messages: ChatMessage[] = [
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              type: "function",
              id: "call1",
              function: {
                name: "test_function",
                arguments: '{"arg": "value"}',
              },
              extra_content: {
                google: {
                  thought_signature: "actual_signature_123",
                },
              },
            },
          ],
        },
      ];

      const options: CompletionOptions = {
        model: "gemini-2.5-pro",
      };

      const body = mockGemini.prepareBody(messages, options, false, true);

      const assistantMsg = body.contents[0];
      expect(assistantMsg.parts.length).toBeGreaterThan(0);

      // Find the function call part
      const functionCallPart = assistantMsg.parts.find(
        (p: any) => p.functionCall,
      );
      expect(functionCallPart).toBeDefined();
      expect(functionCallPart).toHaveProperty("thoughtSignature");
      expect((functionCallPart as any).thoughtSignature).toBe(
        "actual_signature_123",
      );
    });

    it("should use fallback signature when thought_signature is missing", () => {
      const messages: ChatMessage[] = [
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              type: "function",
              id: "call1",
              function: {
                name: "test_function",
                arguments: "{}",
              },
              // No extra_content
            },
          ],
        },
      ];

      const options: CompletionOptions = {
        model: "gemini-2.5-pro",
      };

      const body = mockGemini.prepareBody(messages, options, false, true);

      const assistantMsg = body.contents[0];
      const functionCallPart = assistantMsg.parts.find(
        (p: any) => p.functionCall,
      );

      expect(functionCallPart).toBeDefined();
      expect(functionCallPart).toHaveProperty("thoughtSignature");
      expect((functionCallPart as any).thoughtSignature).toBe(
        "skip_thought_signature_validator",
      );
    });

    it("should only attach thought signature to the first tool call", () => {
      const messages: ChatMessage[] = [
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              type: "function",
              id: "call1",
              function: {
                name: "first_function",
                arguments: "{}",
              },
            },
            {
              type: "function",
              id: "call2",
              function: {
                name: "second_function",
                arguments: "{}",
              },
            },
          ],
        },
      ];

      const options: CompletionOptions = {
        model: "gemini-2.5-pro",
      };

      const body = mockGemini.prepareBody(messages, options, false, true);

      const assistantMsg = body.contents[0];
      const functionCallParts = assistantMsg.parts.filter(
        (p: any) => p.functionCall,
      );

      // Should have 2 function call parts
      expect(functionCallParts.length).toBe(2);

      // First should have signature
      expect(functionCallParts[0]).toHaveProperty("thoughtSignature");

      // Second should NOT have signature
      expect(functionCallParts[1]).not.toHaveProperty("thoughtSignature");
    });

    it("should preserve existing thought signature from tool call", () => {
      const messages: ChatMessage[] = [
        {
          role: "user",
          content: "Call a function",
        },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              type: "function",
              id: "call1",
              function: {
                name: "test_func",
                arguments: "{}",
              },
              extra_content: {
                google: {
                  thought_signature: "preserved_signature",
                },
              },
            },
          ],
        },
        {
          role: "tool",
          toolCallId: "call1",
          content: "Function result",
        },
      ];

      const options: CompletionOptions = {
        model: "gemini-2.5-pro",
      };

      const body = mockGemini.prepareBody(messages, options, false, true);

      // Find the assistant message with tool call
      const assistantMsg = body.contents.find(
        (msg: any) =>
          msg.role === "model" && msg.parts.some((p: any) => p.functionCall),
      );

      expect(assistantMsg).toBeDefined();
      const functionCallPart = assistantMsg.parts.find(
        (p: any) => p.functionCall,
      );

      expect(functionCallPart).toHaveProperty("thoughtSignature");
      expect((functionCallPart as any).thoughtSignature).toBe(
        "preserved_signature",
      );
    });

    it("should handle messages without tool calls", () => {
      const messages: ChatMessage[] = [
        {
          role: "user",
          content: "Hello",
        },
        {
          role: "assistant",
          content: "Hi there!",
        },
      ];

      const options: CompletionOptions = {
        model: "gemini-2.5-pro",
      };

      const body = mockGemini.prepareBody(messages, options, false, true);

      expect(body.contents.length).toBe(2);
      // No function calls, so no thought signatures
      body.contents.forEach((msg: any) => {
        msg.parts.forEach((part: any) => {
          expect(part).not.toHaveProperty("thoughtSignature");
        });
      });
    });
  });

  describe("processGeminiResponse", () => {
    it("should extract thought signature from function call in response", async () => {
      const mockStream = (async function* () {
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      id: "call_123",
                      name: "test_function",
                      args: { arg: "value" },
                    },
                    thoughtSignature: "response_signature_789",
                  },
                ],
              },
            },
          ],
        });
      })();

      const messages: ChatMessage[] = [];
      for await (const message of mockGemini.processGeminiResponse(
        mockStream,
      )) {
        messages.push(message);
      }

      expect(messages.length).toBe(1);
      expect(messages[0].role).toBe("assistant");
      expect(messages[0].toolCalls).toBeDefined();
      expect(messages[0].toolCalls?.length).toBe(1);

      const toolCall = messages[0].toolCalls![0];
      expect((toolCall as any).extra_content?.google?.thought_signature).toBe(
        "response_signature_789",
      );
    });

    it("should handle response without thought signature", async () => {
      const mockStream = (async function* () {
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      id: "call_456",
                      name: "another_function",
                      args: {},
                    },
                  },
                ],
              },
            },
          ],
        });
      })();

      const messages: ChatMessage[] = [];
      for await (const message of mockGemini.processGeminiResponse(
        mockStream,
      )) {
        messages.push(message);
      }

      expect(messages.length).toBe(1);
      expect(messages[0].toolCalls?.length).toBe(1);

      const toolCall = messages[0].toolCalls![0];
      // Should not have extra_content if no signature was present
      expect(
        (toolCall as any).extra_content?.google?.thought_signature,
      ).toBeUndefined();
    });

    it("should handle text response without thought signature", async () => {
      const mockStream = (async function* () {
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: "Just a regular text response",
                  },
                ],
              },
            },
          ],
        });
      })();

      const messages: ChatMessage[] = [];
      for await (const message of mockGemini.processGeminiResponse(
        mockStream,
      )) {
        messages.push(message);
      }

      expect(messages.length).toBe(1);
      expect(messages[0].role).toBe("assistant");
      expect(messages[0].content).toEqual([
        { type: "text", text: "Just a regular text response" },
      ]);
      expect(messages[0].toolCalls).toBeUndefined();
    });

    it("should handle streaming with multiple chunks", async () => {
      const mockStream = (async function* () {
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [{ text: "First " }],
              },
            },
          ],
        });
        yield ",";
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [{ text: "Second" }],
              },
            },
          ],
        });
      })();

      const messages: ChatMessage[] = [];
      for await (const message of mockGemini.processGeminiResponse(
        mockStream,
      )) {
        messages.push(message);
      }

      expect(messages.length).toBe(2);
      expect(messages[0].content).toEqual([{ type: "text", text: "First " }]);
      expect(messages[1].content).toEqual([{ type: "text", text: "Second" }]);
    });

    it("should handle malformed JSON gracefully", async () => {
      const mockStream = (async function* () {
        yield '{"incomplete":';
        yield JSON.stringify({
          candidates: [
            {
              content: {
                parts: [{ text: "Complete" }],
              },
            },
          ],
        });
      })();

      const messages: ChatMessage[] = [];
      for await (const message of mockGemini.processGeminiResponse(
        mockStream,
      )) {
        messages.push(message);
      }

      // Should still process the valid JSON chunk
      expect(messages.length).toBeGreaterThan(0);
    });
  });

  describe("continuePartToGeminiPart", () => {
    it("should convert text parts correctly", () => {
      const textPart = {
        type: "text" as const,
        text: "Hello world",
      };

      const result = mockGemini.continuePartToGeminiPart(textPart);

      expect(result).toEqual({
        text: "Hello world",
      });
    });

    it("should handle image parts with data URLs", () => {
      const imagePart = {
        type: "imageUrl" as const,
        imageUrl: {
          url: "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        },
      };

      const result = mockGemini.continuePartToGeminiPart(imagePart);

      expect(result).toHaveProperty("inlineData");
      expect((result as any).inlineData.mimeType).toBe("image/jpeg");
      expect((result as any).inlineData.data).toBe("/9j/4AAQSkZJRg==");
    });
  });
});
