import { describe, expect, it, vi } from "vitest";
import { GeminiApi } from "./Gemini.js";

describe("GeminiApi thought signature support", () => {
  const mockConfig = {
    apiKey: "test-key",
  };

  describe("_convertBody", () => {
    it("should attach thought signature to first tool call when present in extra_content", () => {
      const api = new GeminiApi(mockConfig);
      const body = {
        model: "gemini-2.5-pro",
        messages: [
          {
            role: "assistant" as const,
            tool_calls: [
              {
                id: "call1",
                type: "function" as const,
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
        ],
      };

      const result = api._convertBody(
        body as any,
        "https://generativelanguage.googleapis.com/v1beta/",
        true,
      );

      expect(result.contents[0].parts[0]).toHaveProperty("thoughtSignature");
      expect(result.contents[0].parts[0].thoughtSignature).toBe(
        "actual_signature_123",
      );
    });

    it("should use fallback signature when thought_signature is missing", () => {
      const api = new GeminiApi(mockConfig);
      const body = {
        model: "gemini-2.5-pro",
        messages: [
          {
            role: "assistant" as const,
            tool_calls: [
              {
                id: "call1",
                type: "function" as const,
                function: {
                  name: "test_function",
                  arguments: '{"arg": "value"}',
                },
                // No extra_content
              },
            ],
          },
        ],
      };

      const result = api._convertBody(
        body as any,
        "https://generativelanguage.googleapis.com/v1beta/",
        true,
      );

      expect(result.contents[0].parts[0]).toHaveProperty("thoughtSignature");
      expect(result.contents[0].parts[0].thoughtSignature).toBe(
        "skip_thought_signature_validator",
      );
    });

    it("should use fallback signature when thought_signature is empty string", () => {
      const api = new GeminiApi(mockConfig);
      const body = {
        model: "gemini-2.5-pro",
        messages: [
          {
            role: "assistant" as const,
            tool_calls: [
              {
                id: "call1",
                type: "function" as const,
                function: {
                  name: "test_function",
                  arguments: '{"arg": "value"}',
                },
                extra_content: {
                  google: {
                    thought_signature: "",
                  },
                },
              },
            ],
          },
        ],
      };

      const result = api._convertBody(
        body as any,
        "https://generativelanguage.googleapis.com/v1beta/",
        true,
      );

      expect(result.contents[0].parts[0]).toHaveProperty("thoughtSignature");
      expect(result.contents[0].parts[0].thoughtSignature).toBe(
        "skip_thought_signature_validator",
      );
    });

    it("should only attach thought signature to first tool call in array", () => {
      const api = new GeminiApi(mockConfig);
      const body = {
        model: "gemini-2.5-pro",
        messages: [
          {
            role: "assistant" as const,
            tool_calls: [
              {
                id: "call1",
                type: "function" as const,
                function: {
                  name: "first_function",
                  arguments: '{"arg": "value1"}',
                },
              },
              {
                id: "call2",
                type: "function" as const,
                function: {
                  name: "second_function",
                  arguments: '{"arg": "value2"}',
                },
              },
            ],
          },
        ],
      };

      const result = api._convertBody(
        body as any,
        "https://generativelanguage.googleapis.com/v1beta/",
        true,
      );

      // First call should have thought signature
      expect(result.contents[0].parts[0]).toHaveProperty("thoughtSignature");
      expect(result.contents[0].parts[0].thoughtSignature).toBe(
        "skip_thought_signature_validator",
      );

      // Second call should NOT have thought signature
      expect(result.contents[0].parts[1]).not.toHaveProperty(
        "thoughtSignature",
      );
    });

    it("should handle tool calls in messages correctly", () => {
      const api = new GeminiApi(mockConfig);
      const body = {
        model: "gemini-2.5-pro",
        messages: [
          {
            role: "user" as const,
            content: "Call a function",
          },
          {
            role: "assistant" as const,
            tool_calls: [
              {
                id: "call1",
                type: "function" as const,
                function: {
                  name: "test_func",
                  arguments: "{}",
                },
                extra_content: {
                  google: {
                    thought_signature: "sig_from_previous_call",
                  },
                },
              },
            ],
          },
          {
            role: "tool" as const,
            tool_call_id: "call1",
            content: "Function result",
          },
        ],
      };

      const result = api._convertBody(
        body as any,
        "https://generativelanguage.googleapis.com/v1beta/",
        true,
      );

      // Should have user message, assistant with tool call, and tool response
      expect(result.contents.length).toBe(3);

      // Tool call should have the signature
      expect(result.contents[1].parts[0]).toHaveProperty("thoughtSignature");
      expect(result.contents[1].parts[0].thoughtSignature).toBe(
        "sig_from_previous_call",
      );
    });
  });

  describe("handleStreamResponse", () => {
    function createMockResponse(text: string): any {
      const encoder = new TextEncoder();
      const encoded = encoder.encode(text);
      let index = 0;

      return {
        status: 200,
        text: () => Promise.resolve(text),
        body: {
          getReader: () => ({
            read: async () => {
              if (index === 0) {
                index++;
                return { done: false, value: encoded };
              }
              return { done: true, value: undefined };
            },
          }),
          [Symbol.asyncIterator]: async function* () {
            yield encoded;
          },
        },
      };
    }

    it("should extract and emit thought signature from text parts", async () => {
      const api = new GeminiApi(mockConfig);

      // Mock response with thoughtSignature
      const mockResponse = createMockResponse(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: "Hello",
                    thoughtSignature: "stream_signature_123",
                  },
                ],
              },
            },
          ],
        }),
      );

      const chunks: any[] = [];
      for await (const chunk of api.handleStreamResponse(
        mockResponse,
        "gemini-2.5-pro",
      )) {
        chunks.push(chunk);
      }

      // Should have 2 chunks: one with thought signature, one with text
      expect(chunks.length).toBeGreaterThanOrEqual(2);

      // First chunk should contain thought signature in extra_content
      const signatureChunk = chunks.find(
        (c) => c.choices[0]?.delta?.extra_content?.google?.thought_signature,
      );
      expect(signatureChunk).toBeDefined();
      expect(
        signatureChunk.choices[0].delta.extra_content.google.thought_signature,
      ).toBe("stream_signature_123");

      // Should also have text chunk
      const textChunk = chunks.find(
        (c) => c.choices[0]?.delta?.content === "Hello",
      );
      expect(textChunk).toBeDefined();
    });

    it("should extract and emit thought signature from function call parts", async () => {
      const api = new GeminiApi(mockConfig);

      const mockResponse = createMockResponse(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      id: "call_id",
                      name: "test_func",
                      args: { arg: "value" },
                    },
                    thoughtSignature: "func_signature_456",
                  },
                ],
              },
            },
          ],
        }),
      );

      const chunks: any[] = [];
      for await (const chunk of api.handleStreamResponse(
        mockResponse,
        "gemini-2.5-pro",
      )) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);

      // Find the chunk with tool call
      const toolCallChunk = chunks.find((c) => c.choices[0]?.delta?.tool_calls);
      expect(toolCallChunk).toBeDefined();

      const toolCall = toolCallChunk.choices[0].delta.tool_calls[0];
      expect(toolCall).toBeDefined();
      expect(toolCall.extra_content?.google?.thought_signature).toBe(
        "func_signature_456",
      );
    });

    it("should handle response without thought signature", async () => {
      const api = new GeminiApi(mockConfig);

      const mockResponse = createMockResponse(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: "Response without signature",
                  },
                ],
              },
            },
          ],
        }),
      );

      const chunks: any[] = [];
      for await (const chunk of api.handleStreamResponse(
        mockResponse,
        "gemini-2.5-pro",
      )) {
        chunks.push(chunk);
      }

      // Should only have text chunk, no signature chunk
      expect(chunks.length).toBe(1);
      expect(chunks[0].choices[0].delta.content).toBe(
        "Response without signature",
      );

      // Should not have extra_content with thought signature
      expect(
        chunks[0].choices[0].delta.extra_content?.google?.thought_signature,
      ).toBeUndefined();
    });

    it("should handle streaming with usage metadata", async () => {
      const api = new GeminiApi(mockConfig);

      const mockResponse = createMockResponse(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: "Text part",
                    thoughtSignature: "sig_with_usage",
                  },
                ],
              },
            },
          ],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 20,
            totalTokenCount: 30,
          },
        }),
      );

      const chunks: any[] = [];
      for await (const chunk of api.handleStreamResponse(
        mockResponse,
        "gemini-2.5-pro",
      )) {
        chunks.push(chunk);
      }

      // Should have signature chunk, text chunk, and usage chunk
      expect(chunks.length).toBeGreaterThanOrEqual(2);

      // Find usage chunk
      const usageChunk = chunks.find((c) => c.usage);
      expect(usageChunk).toBeDefined();
      expect(usageChunk.usage.total_tokens).toBe(30);
    });
  });
});
