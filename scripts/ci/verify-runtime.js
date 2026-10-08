const assert = require("node:assert/strict");
const { randomBytes } = require("node:crypto");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");

const ProxyServerSystem = require("../../src/core/ProxyServerSystem");

async function verifyRuntime() {
    const originalDirectory = process.cwd();
    const isolatedDirectory = await fs.mkdtemp(path.join(os.tmpdir(), "aistudio-runtime-"));
    const deadline = setTimeout(() => {
        console.error("Runtime verification timed out");
        process.exit(1);
    }, 90000);
    deadline.unref();

    let server;
    let context;
    try {
        await fs.mkdir(path.join(isolatedDirectory, "configs", "auth"), { recursive: true });
        await fs.copyFile(
            path.resolve(__dirname, "../../configs/models.json"),
            path.join(isolatedDirectory, "configs", "models.json")
        );
        process.chdir(isolatedDirectory);
        process.env.HOST = "127.0.0.1";
        process.env.API_KEYS = randomBytes(24).toString("hex");
        process.env.CHECK_UPDATE = "false";
        process.env.ENABLE_USAGE_STATS = "false";
        for (const key of ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]) {
            delete process.env[key];
        }

        server = new ProxyServerSystem();
        server.config.httpPort = 0;
        server.config.wsPort = 0;
        assert.equal(server.authSource.availableIndices.length, 0);
        await server.start();
        assert.ok(server.wsServer.address().port > 0);

        const baseUrl = `http://127.0.0.1:${server.httpServer.address().port}`;
        const response = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(10000) });
        assert.equal(response.status, 200);
        assert.equal((await response.json()).status, "ok");

        await server.browserManager._ensureBrowser();
        assert.ok(server.browserManager.browser.isConnected());
        context = await server.browserManager.browser.newContext();
        const page = await context.newPage();
        const pageErrors = [];
        page.on("pageerror", error => pageErrors.push(error.message));
        const loginResponse = await page.goto(`${baseUrl}/login`, { timeout: 20000 });
        assert.equal(loginResponse.status(), 200);
        await page.locator(".login-form input[type=password]").waitFor({ state: "visible", timeout: 15000 });
        assert.deepEqual(pageErrors, []);

        const browserHealth = await page.request.get(`${baseUrl}/health`);
        assert.equal(browserHealth.status(), 200);
        assert.equal((await browserHealth.json()).browserConnected, true);
        console.log(
            `Runtime verification passed: Playwright ${require("playwright/package.json").version}, ` +
                "Camoufox, HTTP/WebSocket listeners and frontend login page"
        );
    } finally {
        try {
            if (context) await context.close();
        } finally {
            try {
                if (server) await server.shutdown();
            } finally {
                clearTimeout(deadline);
                process.chdir(originalDirectory);
                await fs.rm(isolatedDirectory, { recursive: true, force: true });
            }
        }
    }
}

verifyRuntime().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
