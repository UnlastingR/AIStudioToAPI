/**
 * File: src/core/FormatConverter/EmbeddingsConverter.js
 * Description: OpenAI Embeddings request conversion.
 *
 * Author: Ellinav, iBenzene, bbbugg
 */

const FormatConverter = require("./CommonConverter");

class EmbeddingsConverter extends FormatConverter {
    /**
     * Convert an OpenAI embeddings request to Gemini native batchEmbedContents format.
     *
     * The Google OpenAI-compatible embeddings endpoint requires a Gemini API key and cannot be
     * called with the browser session used by this project. Use the native endpoint instead and
     * convert the response back to OpenAI format.
     *
     * @param {object} openaiBody - OpenAI embeddings request body
     * @returns {{
     *   cleanModelName: string,
     *   encodingFormat: "float"|"base64",
     *   googleRequest: object,
     *   path: string
     * }}
     */
    translateOpenAIEmbeddingsToGoogle(openaiBody) {
        this.logger.debug("[Adapter] Starting translation of OpenAI embeddings request to Gemini native format...");

        if (!openaiBody || typeof openaiBody !== "object" || Array.isArray(openaiBody)) {
            throw new Error("Embeddings request body must be an object.");
        }

        const rawModelName = typeof openaiBody.model === "string" ? openaiBody.model.trim() : "";
        const cleanModelName = rawModelName.replace(/^models\//, "");
        if (!cleanModelName || !/^[a-zA-Z0-9._-]+$/.test(cleanModelName)) {
            throw new Error("Embeddings request must include a valid model.");
        }

        const inputs = typeof openaiBody.input === "string" ? [openaiBody.input] : openaiBody.input;
        if (!Array.isArray(inputs) || inputs.length === 0 || !inputs.every(input => typeof input === "string")) {
            throw new Error("Embeddings input must be a string or a non-empty array of strings.");
        }

        const encodingFormat = openaiBody.encoding_format || "float";
        if (encodingFormat !== "float" && encodingFormat !== "base64") {
            throw new Error('Embeddings encoding_format must be either "float" or "base64".');
        }

        let outputDimensionality = null;
        if (openaiBody.dimensions !== undefined) {
            outputDimensionality = Number(openaiBody.dimensions);
            if (!Number.isInteger(outputDimensionality) || outputDimensionality < 1) {
                throw new Error("Embeddings dimensions must be a positive integer.");
            }
        }

        const googleRequest = {
            requests: inputs.map(input => ({
                content: {
                    parts: [{ text: input }],
                },
                model: `models/${cleanModelName}`,
                ...(outputDimensionality !== null ? { outputDimensionality } : {}),
            })),
        };
        const path = `/v1beta/models/${cleanModelName}:batchEmbedContents`;

        this.logger.debug(
            `[Adapter] Debug: Final Gemini Native Embeddings Body = ${JSON.stringify(googleRequest, null, 2)}`
        );
        this.logger.debug(`[Adapter] Debug: Final Gemini Native Embeddings Path = ${path}`);
        this.logger.debug("[Adapter] OpenAI embeddings to Gemini native translation complete.");

        return { cleanModelName, encodingFormat, googleRequest, path };
    }

    /**
     * Convert a Gemini native batchEmbedContents response to OpenAI embeddings format.
     * @param {Buffer|string|object} googleResponseBody - Gemini batch embedding response
     * @param {string} modelName - Model name returned to the OpenAI client
     * @param {"float"|"base64"} [encodingFormat="float"] - OpenAI embedding encoding
     * @returns {Buffer} OpenAI embeddings response
     */
    translateGoogleEmbeddingsToOpenAI(googleResponseBody, modelName, encodingFormat = "float") {
        let googleResponse;
        if (Buffer.isBuffer(googleResponseBody)) {
            googleResponse = JSON.parse(googleResponseBody.toString());
        } else if (typeof googleResponseBody === "string") {
            googleResponse = JSON.parse(googleResponseBody);
        } else {
            googleResponse = googleResponseBody;
        }

        if (!googleResponse || !Array.isArray(googleResponse.embeddings)) {
            throw new Error("Backend batchEmbedContents response did not contain embeddings.");
        }

        const data = googleResponse.embeddings.map((embedding, index) => {
            if (!Array.isArray(embedding?.values)) {
                throw new Error(`Backend embedding at index ${index} did not contain values.`);
            }

            let values = embedding.values;
            if (encodingFormat === "base64") {
                const buffer = Buffer.allocUnsafe(values.length * Float32Array.BYTES_PER_ELEMENT);
                values.forEach((value, valueIndex) => {
                    buffer.writeFloatLE(Number(value), valueIndex * Float32Array.BYTES_PER_ELEMENT);
                });
                values = buffer.toString("base64");
            }

            return {
                embedding: values,
                index,
                object: "embedding",
            };
        });

        const promptTokens = Number(googleResponse.usageMetadata?.promptTokenCount ?? googleResponse.tokenCount ?? 0);
        const totalTokens = Number(googleResponse.usageMetadata?.totalTokenCount ?? promptTokens);

        return Buffer.from(
            JSON.stringify({
                data,
                model: modelName,
                object: "list",
                usage: {
                    prompt_tokens: Number.isFinite(promptTokens) ? promptTokens : 0,
                    total_tokens: Number.isFinite(totalTokens) ? totalTokens : 0,
                },
            })
        );
    }
}

module.exports = EmbeddingsConverter;
