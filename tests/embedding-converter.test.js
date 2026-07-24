const assert = require("node:assert/strict");
const test = require("node:test");

const FormatConverter = require("../src/core/FormatConverter");

const logger = {
    debug() {},
    error() {},
    info() {},
    warn() {},
};

const converter = new FormatConverter(logger, {});

test("converts a single OpenAI embedding input to Gemini batchEmbedContents", () => {
    const result = converter.translateOpenAIEmbeddingsToGoogle({
        input: "hello",
        model: "gemini-embedding-2",
    });

    assert.equal(result.cleanModelName, "gemini-embedding-2");
    assert.equal(result.encodingFormat, "float");
    assert.equal(result.path, "/v1beta/models/gemini-embedding-2:batchEmbedContents");
    assert.deepEqual(result.googleRequest, {
        requests: [
            {
                content: {
                    parts: [{ text: "hello" }],
                },
                model: "models/gemini-embedding-2",
            },
        ],
    });
});

test("converts multiple inputs and dimensions", () => {
    const result = converter.translateOpenAIEmbeddingsToGoogle({
        dimensions: 768,
        encoding_format: "base64",
        input: ["first", "second"],
        model: "models/gemini-embedding-2",
    });

    assert.equal(result.encodingFormat, "base64");
    assert.equal(result.googleRequest.requests.length, 2);
    assert.equal(result.googleRequest.requests[0].outputDimensionality, 768);
    assert.equal(result.googleRequest.requests[1].content.parts[0].text, "second");
});

test("converts Gemini float embeddings to OpenAI format", () => {
    const response = converter.translateGoogleEmbeddingsToOpenAI(
        {
            embeddings: [{ values: [0.25, -0.5] }, { values: [0.75, 1] }],
            usageMetadata: {
                promptTokenCount: 7,
            },
        },
        "gemini-embedding-2"
    );
    const parsed = JSON.parse(response.toString());

    assert.equal(parsed.object, "list");
    assert.equal(parsed.model, "gemini-embedding-2");
    assert.deepEqual(parsed.data, [
        {
            embedding: [0.25, -0.5],
            index: 0,
            object: "embedding",
        },
        {
            embedding: [0.75, 1],
            index: 1,
            object: "embedding",
        },
    ]);
    assert.deepEqual(parsed.usage, {
        prompt_tokens: 7,
        total_tokens: 7,
    });
});

test("converts Gemini embeddings to OpenAI base64 float32 encoding", () => {
    const response = converter.translateGoogleEmbeddingsToOpenAI(
        {
            embeddings: [{ values: [0.25, -0.5] }],
            tokenCount: "2",
        },
        "gemini-embedding-2",
        "base64"
    );
    const parsed = JSON.parse(response.toString());
    const decoded = Buffer.from(parsed.data[0].embedding, "base64");

    assert.equal(decoded.length, 8);
    assert.equal(decoded.readFloatLE(0), 0.25);
    assert.equal(decoded.readFloatLE(4), -0.5);
    assert.deepEqual(parsed.usage, {
        prompt_tokens: 2,
        total_tokens: 2,
    });
});

test("rejects unsupported OpenAI embedding inputs", () => {
    assert.throws(
        () =>
            converter.translateOpenAIEmbeddingsToGoogle({
                input: [1, 2, 3],
                model: "gemini-embedding-2",
            }),
        /string or a non-empty array of strings/
    );
    assert.throws(
        () =>
            converter.translateOpenAIEmbeddingsToGoogle({
                encoding_format: "binary",
                input: "hello",
                model: "gemini-embedding-2",
            }),
        /encoding_format/
    );
});
