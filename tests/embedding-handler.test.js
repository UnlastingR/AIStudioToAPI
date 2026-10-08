const assert = require("node:assert/strict");
const test = require("node:test");

const FormatConverter = require("../src/core/FormatConverter");
const RequestHandler = require("../src/core/RequestHandler");

const logger = { debug() {}, error() {}, info() {}, warn() {} };

function createHandler() {
    const handler = Object.create(RequestHandler.prototype);
    handler.logger = logger;
    handler.formatConverter = new FormatConverter(logger, {});
    handler.authSwitcher = { currentAuthIndex: 0, failureCount: 0 };
    handler.timeouts = { FAKE_STREAM: 1000 };
    handler._generateRequestId = () => "embedding-request";
    handler._ensureBrowserBackedRequestReady = async () => true;
    handler._initializeProxyRequestAttempt = request => {
        request.request_attempt_id = "embedding-attempt";
    };
    for (const name of [
        "_startTrackedRequest",
        "_setResponseApiFormat",
        "_updateTrackedRequest",
        "_setupClientDisconnectHandler",
        "_cleanupRequestResources",
        "_finalizeTrackedRequest",
        "_logGeminiNativeResponseDebug",
        "_setResponseHeaders",
    ]) {
        handler[name] = () => {};
    }
    handler._sendErrorResponse = (res, status, message, type) => {
        res.error = { status, message, type };
    };
    handler._handleRequestError = error => {
        throw error;
    };
    return handler;
}

test("modular handler sends native embedding requests without forwarding client credentials", async () => {
    const handler = createHandler();
    let dispatched;
    handler.connectionRegistry = { createMessageQueue: () => ({}) };
    handler._handleNonStreamResponse = async request => {
        dispatched = request;
    };

    await handler.processOpenAIEmbeddingsRequest(
        {
            body: { input: "hello", model: "models/gemini-embedding-2", encoding_format: "base64" },
            headers: { authorization: "Bearer client-test-key" },
            query: { key: "client-test-key" },
        },
        {}
    );

    assert.equal(dispatched.path, "/v1beta/models/gemini-embedding-2:batchEmbedContents");
    assert.deepEqual(dispatched.headers, { "Content-Type": "application/json" });
    assert.deepEqual(dispatched.query_params, {});
    assert.equal(dispatched.response_transform, "batchEmbedToOpenAI");
    assert.equal(dispatched.response_model, "gemini-embedding-2");
    assert.equal(dispatched.response_encoding_format, "base64");
    assert.equal(JSON.parse(dispatched.body).requests[0].content.parts[0].text, "hello");
});

test("modular handler rejects invalid embedding input with HTTP 400", async () => {
    const handler = createHandler();
    const response = {};
    await handler.processOpenAIEmbeddingsRequest(
        { body: { input: [1, 2], model: "gemini-embedding-2" } },
        response
    );
    assert.equal(response.error.status, 400);
    assert.equal(response.error.type, "invalid_request_error");
});

test("modular response handler returns OpenAI embeddings from native Gemini chunks", async () => {
    const handler = createHandler();
    const chunks = [
        { event_type: "chunk", data: JSON.stringify({ embeddings: [{ values: [0.25, -0.5] }], tokenCount: 2 }) },
        { type: "STREAM_END" },
    ];
    handler._executeRequestWithRetries = async () => ({
        success: true,
        message: {},
        queue: { dequeue: async () => chunks.shift() },
    });
    const response = {
        get: () => null,
        type(value) { this.contentType = value; },
        send(value) { this.body = value; },
    };
    await handler._handleNonStreamResponse(
        {
            request_id: "embedding-request",
            is_generative: false,
            response_transform: "batchEmbedToOpenAI",
            response_model: "gemini-embedding-2",
            response_encoding_format: "base64",
        },
        {},
        {},
        response
    );

    const body = JSON.parse(response.body.toString());
    const decoded = Buffer.from(body.data[0].embedding, "base64");
    assert.equal(response.contentType, "application/json");
    assert.equal(body.object, "list");
    assert.equal(body.model, "gemini-embedding-2");
    assert.equal(decoded.readFloatLE(0), 0.25);
    assert.equal(decoded.readFloatLE(4), -0.5);
    assert.deepEqual(body.usage, { prompt_tokens: 2, total_tokens: 2 });
});
