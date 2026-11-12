# Remote MCP and REST security (Phase 5)

How to run the Customer Support Assistant as a **remote MCP server** (Streamable HTTP) and
REST/WebSocket API with authentication, and how to connect it to Claude and ChatGPT.

Vendor UIs and the MCP authorization spec change often. Each external claim below gives its
source and the date it was checked. **Last checked: 2026-11-11.**

## 1. Surfaces at a glance

| Command | Transport | Auth |
|---|---|---|
| `python app.py --mode mcp` | MCP over stdio | **None.** The MCP client launched the process; the spec says stdio "SHOULD NOT" use HTTP auth |
| `python app.py --mode mcp --transport http` | MCP Streamable HTTP at `http://BIND_HOST:MCP_HTTP_PORT/mcp` (default `127.0.0.1:8001`) | Required (`AUTH_MODE`) |
| `python app.py --mode api` | REST + WebSocket at `BIND_HOST:PORT` (default `127.0.0.1:8000`) | Required on every route except `GET /health` |

`--host` / `--port` override `BIND_HOST` / `PORT` / `MCP_HTTP_PORT`. Tool names and result
contracts are the same over stdio and HTTP. The tool list still depends on `LLM_MODE`
(`answer_question` only in `api` / `offline`).

**MCP SDK version.** The installed `mcp` 1.30 (pin `mcp>=1.10,<2`) is enough. Its `FastMCP`
provides Streamable HTTP (protocol revisions up to `2025-11-25`) and `TransportSecuritySettings`
(Host/Origin checks). Its session manager binds each MCP session to the authenticated principal
in `scope["user"]`: a session id used with another credential gets 404. Authentication itself
is our own ASGI middleware (`api/security.py`) in front of the SDK app, so the SDK's
`AuthSettings` / `TokenVerifier` are not used and there was no reason to migrate to mcp 2.x
`MCPServer`.

## 2. Authentication modes (`AUTH_MODE`)

| Mode | Use it for | Needs |
|---|---|---|
| `token` (default) | Single tenant: your own agents, CRM, widget backend, Claude "request headers" (beta) | `API_KEYS=name:token,...` (tokens ≥ 32 URL-safe chars) |
| `oauth` | Claude custom connector with sign-in, **ChatGPT** (OAuth only) | `OAUTH_ISSUER`, `OAUTH_RESOURCE_URL`, `OAUTH_JWKS_URL` |
| `both` | Both of the above at once | everything above |
| `none` | Local experiments only | a loopback bind; refused otherwise |

The server **fails closed**: a mode without its settings, a short or duplicate key, an HS/`none`
algorithm, a non-https issuer/JWKS URL for a remote host, or `AUTH_MODE=none` on a non-loopback
bind (or with a public `ALLOWED_HOSTS` / `OAUTH_RESOURCE_URL`) prints a JSON error on stderr
and exits with status 2 before listening.

Generate a static key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# .env:  API_KEYS=crm:<that value>,widget:<another value>
```

Clients send `Authorization: Bearer <token>` on **every** HTTP request, including the
WebSocket handshake. Only SHA-256 digests of the keys are kept in memory and comparison is
constant-time. The key **name** appears in logs and keys the rate limit; the token never
appears in either.

### OAuth 2.1 resource-server mode

This server is only the **resource server**. It never runs an authorization server, never
issues tokens and never handles DCR, CIMD or PKCE itself; those belong to your authorization
server (Keycloak, Auth0, Entra ID, Okta, …). Per request it checks:

- a JWS access token (JWT) whose `alg` is in `OAUTH_ALGORITHMS`. Only asymmetric algorithms
  are allowed (RS*/PS*/ES*/EdDSA). `none` and `HS*` are refused, which closes the
  algorithm-confusion attack. A `crit` header is refused.
- a signing key from `OAUTH_JWKS_URL` only (`jku`/`x5u` are ignored), matched by `kid`. Key
  type and curve must fit the algorithm; private-key JWKs and RSA keys under 2048 bits are
  refused. The JWKS is fetched over https (http only for local hosts) with no redirects, a
  size cap and a timeout. It is cached for `OAUTH_JWKS_CACHE_TTL` and refreshed for an
  unknown `kid` at most once every 30 s. When no fresh key set can be fetched, requests get
  503, not 200.
- the `exp`, `iss` and `aud` claims are all required. `iss` must equal `OAUTH_ISSUER` exactly.
  `aud` must contain `OAUTH_RESOURCE_URL` (RFC 8707 audience binding; a trailing `/` is
  tolerated). A token minted for any other resource is refused.
- every scope in `OAUTH_REQUIRED_SCOPES` is present in `scope` (space-separated) or `scp`.
  If not, the answer is 403 `insufficient_scope` with a `scope=` challenge.

Opaque tokens and RFC 7662 introspection are **not supported**: configure the authorization
server to issue JWT access tokens for this resource.

What the authorization server must do for the vendors below: support authorization code +
PKCE S256 and the `resource` parameter (put it into `aud`), publish RFC 8414 or OIDC discovery
metadata, and offer CIMD (`client_id_metadata_document_supported: true` plus `none` in
`token_endpoint_auth_methods_supported`) or DCR (`registration_endpoint`), or accept a
pre-registered client. Register the vendor callback URLs from §5.

### What clients see

| Situation | Status | Notes |
|---|---|---|
| `Host` not allowlisted | 421 | DNS-rebinding guard (always on for loopback binds) |
| `Origin` present and not in `ALLOWED_ORIGINS` (incl. `null`) | 403 | applies to open routes too; requests without `Origin` pass |
| `?access_token=` in the URL | 400 | tokens are accepted only in the header |
| no / non-Bearer credentials | 401 | `WWW-Authenticate: Bearer realm=…[, scope=…][, resource_metadata=…]` |
| invalid, unknown, malformed, expired, wrong-issuer, wrong-audience token | 401 | always the same body: `the access token is invalid or expired` (the reason goes to the server log only) |
| missing scope | 403 | `error="insufficient_scope", scope="…"` |
| key set unreachable | 503 + `Retry-After` | fail closed |
| over the rate limit | 429 + `Retry-After` | per key; per client IP for open routes and failed sign-ins |
| WebSocket refused | handshake answered 403 (close 1008); 1013 when rate limited mid-stream | |

In `oauth`/`both` mode the RFC 9728 Protected Resource Metadata is served openly at
`/.well-known/oauth-protected-resource/<resource path>` (e.g. `…/mcp`) and at the root
`/.well-known/oauth-protected-resource`. The 401 challenge points at the path-inserted URL.
The REST app serves the same document: one deployment is one resource (`OAUTH_RESOURCE_URL`),
so a token for it works on both surfaces.

**No token passthrough.** The middleware removes the `Authorization` header before any route,
tool or SDK code runs and hands them only the verified principal (no token). The MCP SDK user
object carries an empty token string. The only outbound requests that auth itself makes are
JWKS GETs to the configured URL, without credentials. Tests assert that neither the JWKS fetch
nor the escalation webhook ever carries the caller's token.

### Origins, hosts, proxies, rate limits

- `ALLOWED_ORIGINS`: exact `scheme://host[:port]` entries, no wildcards. Browsers cannot set
  an `Authorization` header on a WebSocket, so browser WS clients are not supported. Use a
  backend that holds the key.
- `ALLOWED_HOSTS`: on a loopback bind `localhost`/`127.0.0.1`/`[::1]` are always accepted. For
  a public deployment set your public host name, or the Host header is not checked (a warning
  is logged at start).
- Behind a reverse proxy or tunnel, terminate TLS there and forward to the loopback bind.
  uvicorn trusts `X-Forwarded-For` only from `127.0.0.1` (its default). The per-IP limit
  therefore uses the address the local proxy appends; a proxy that passes a client-supplied
  header through unchanged would let clients rotate their per-IP key. Per-key limits are
  unaffected: they key on the verified credential.
- Rate limits are in memory, per process: each worker has its own counts.

## 3. Offline mode and remote hosting

`LLM_MODE=offline` is incompatible with public hosting by definition (plan §2). At startup the
offline guard refuses:

- a non-local `OAUTH_ISSUER` or `OAUTH_JWKS_URL` (and every other remote endpoint, as before);
- an HTTP surface bound to anything other than loopback, or a private IP listed in
  `OFFLINE_ALLOWED_HOSTS` (an air-gapped LAN). This includes `0.0.0.0` / `::`;
- a public `OAUTH_RESOURCE_URL` or public `ALLOWED_HOSTS` name.

## 4. Getting a public HTTPS URL (needed for `client` mode)

Claude and ChatGPT connect **from the vendor's cloud**, so the MCP endpoint must be reachable
over public HTTPS. Anthropic's outbound range is `160.79.104.0/21` (source: claude.com docs
below). Where to host is **deliberately undecided** (plan §6 #5). For a trial:

```bash
# .env: LLM_MODE=client, AUTH_MODE=token (or oauth), API_KEYS=..., ALLOWED_HOSTS=<tunnel host>
python app.py --mode mcp --transport http            # listens on 127.0.0.1:8001
# then expose it with any HTTPS tunnel (e.g. cloudflared / ngrok) pointing at 127.0.0.1:8001
# connector URL: https://<tunnel host>/mcp
```

With OAuth, set `OAUTH_RESOURCE_URL` to exactly the URL users enter
(`https://<public host>/mcp`). Claude requires the metadata `resource` to equal that URL. Never
run `AUTH_MODE=none` behind a tunnel. The server refuses to start that way when `ALLOWED_HOSTS`
names the tunnel host. A tunnel that forwards its public `Host` header to a loopback bind gets
421 unless `ALLOWED_HOSTS` lists that host. The server still cannot detect a tunnel that
rewrites `Host` to `localhost`.

## 5. Connecting chat apps

### Claude custom connector — verified against official docs on 2026-11-11

Sources: <https://claude.com/docs/connectors/custom> ("Add a connector that isn't in the
directory") and <https://claude.com/docs/connectors/building/authentication>. The older
support.claude.com article URL from the plan now returns 404.

- **Plans:** custom connectors by URL work on Free (one connector), Pro, Max, Ultra, Team and
  Enterprise. On Team/Enterprise an Owner adds it under **Organization settings → Connectors →
  Add → Custom → Web**, and members then **Connect** under **Customize → Connectors**.
- **Individual plans:** **Customize → Connectors → Add → Add custom connector**. Enter a name
  and the server URL (`https://<public host>/mcp`), click **Continue**, choose authentication,
  click **Add**, then **Connect**.
- **With `AUTH_MODE=oauth` (recommended):** choose **Sign in now** (or **Sign in when needed**).
  For **OAuth client** choose **Use Claude's published identity** (CIMD; your authorization
  server must advertise CIMD), **Register automatically** (DCR) or **Use your own OAuth client**
  (client ID registered at your authorization server). Register the redirect URI
  `https://claude.ai/api/mcp/auth_callback` (hosted apps). Claude Code uses a loopback redirect
  on any port (`http://localhost/callback`, `http://127.0.0.1/callback`, port ignored). Claude
  sends PKCE S256 and needs a **401** (not 200) with `resource_metadata` to start sign-in; this
  server sends that. It uses only the **first** `authorization_servers` entry and refreshes
  tokens on 401 and up to 5 minutes before expiry.
- **With `AUTH_MODE=token`:** choose **No sign-in** and add a **Request header**
  `authorization` whose value is `Bearer <token>` (Claude sends the value verbatim, so include
  `Bearer `). Per the docs this "request headers" option is **beta, for a limited set of
  organizations**. If your dialog has no **Request headers** section, use OAuth. All members of
  an organization then share that one key.
- Authentication settings cannot be edited later: remove and re-add the connector.

### ChatGPT — partly verified on 2026-11-11

Verified at <https://developers.openai.com/apps-sdk/build/auth>:

- ChatGPT supports OAuth (or no auth) only. It "does not support machine-to-machine OAuth
  grants" and cannot present custom API keys, **so use `AUTH_MODE=oauth` (or `both`)**.
- It needs Protected Resource Metadata (`resource`, `authorization_servers`, optional
  `scopes_supported`) and a 401 with `WWW-Authenticate: Bearer resource_metadata="…"`. It
  appends `resource` to authorization and token requests, and the server must reject tokens
  without the expected audience or scopes. This server does all of that.
- Client registration: CIMD is preferred (client id `https://chatgpt.com/oauth/client.json`,
  token endpoint auth `none` or `private_key_jwt`), then DCR, or predefined clients.
- Redirect URI: `https://chatgpt.com/connector_platform_oauth_redirect` when your authorization
  server supports RFC 9207 issuer identification. Otherwise it is
  `https://chatgpt.com/connector/oauth/{callback_id}`; copy the exact one from the connector's
  management page.
- Request `offline_access` (if your authorization server offers it) so ChatGPT gets a refresh
  token.

Verified at <https://developers.openai.com/apps-sdk/deploy/connect-chatgpt>: the server needs
a public HTTPS endpoint, entered "including the `/mcp` path". The page describes adding it at
**ChatGPT Plugins (chatgpt.com/plugins) → + → Add custom MCP server** (name, description,
connection method, authentication, accept the risk warning, **Create as a plugin**). After a
redeploy, use **Refresh** on the connection. "A development tunnel or another HTTPS forwarding
service" works for local testing.

**Not verified:** the developer-mode guide
(`developers.openai.com/api/docs/guides/developer-mode`, cited in plan §7) returned 404, and
the help-center article (help.openai.com 12584461) returned 403. So the current **plan
eligibility** for developer mode / custom MCP servers and the exact Settings path are
unconfirmed. Search summaries disagreed: Business/Enterprise/Edu for full (write) access,
possibly read-only for Plus/Pro. Check the in-app UI before relying on it.

## 6. MCP authorization spec — what was checked

Source: <https://modelcontextprotocol.io/specification/latest/basic/authorization> (resolves
to revision **2026-07-28**), plus its *Authorization Server Discovery* and *Client
Registration* pages, checked 2026-11-11.

- MCP servers act as OAuth 2.1 resource servers and **MUST** implement RFC 9728 Protected
  Resource Metadata with at least one `authorization_servers` entry. Discovery is via
  `WWW-Authenticate … resource_metadata` on 401, or the well-known URIs (path-inserted, then
  root). **Done:** both.
- Servers **SHOULD** include `scope` in the challenge. **Done** when `OAUTH_REQUIRED_SCOPES` is
  set. Insufficient scope → 403 with `error="insufficient_scope"`, `scope`, `resource_metadata`.
  **Done.**
- Servers **MUST** validate that tokens were issued for them as audience (RFC 8707), **MUST**
  return 401 for invalid or expired tokens, and "**MUST NOT** accept or transit any other
  tokens". **Done:** `aud` check, Authorization header stripped, no forwarding. Tokens in the
  query string are refused (clients **MUST NOT** send them there).
- Client registration (the authorization server's concern, not this server's): **CIMD is
  SHOULD**. **DCR is MAY and marked deprecated**, kept for backwards compatibility.
  Pre-registration is SHOULD for clients. Client priority: pre-registered → CIMD (if
  advertised) → DCR → manual entry. This resolves the DCR-vs-CIMD disagreement noted in plan
  §7: current spec and both vendors prefer CIMD, and both vendors still fall back to DCR.
- Stdio: implementations "SHOULD NOT follow this specification". **Done:** stdio stays
  unauthenticated.
- The Streamable HTTP requirement that servers validate `Origin` (403 when invalid) is applied
  here; it is also mirrored into the SDK's own transport check.

Note: the vendor auth pages still link to spec revision `2025-11-25`, and the installed SDK
speaks up to `2025-11-25`. The resource-server requirements above are the same in both
revisions.

## 7. Known limits

- No RFC 7662 introspection, so opaque access tokens are not supported. JWT access tokens only.
- Rate limits and JWKS cache are per process (in memory).
- No per-tool scopes: one scope set guards the whole surface. No ChatGPT tool-level
  `securitySchemes` / "mixed auth".
- Static keys are rotated by editing `API_KEYS` and restarting.
- Not exercised live: Claude or ChatGPT actually connecting, any real authorization server
  (Keycloak/Auth0/Entra), and a real tunnel. Locally verified: MCP over Streamable HTTP with
  the official Python MCP client (static token), curl checks (401 / 403 / 421 / 400), OAuth
  mode against a local JWKS server with locally minted JWTs, and REST + WebSocket under
  uvicorn.
