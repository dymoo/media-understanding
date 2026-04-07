FROM node:22-slim AS builder

ARG SUPERGATEWAY_VERSION=3.4.3
ARG YT_DLP_VERSION=2026.03.17
ARG TARGETARCH

ENV XDG_CACHE_HOME=/opt/media-understanding-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    g++ \
    make \
    python3 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json pnpm-lock.yaml tsconfig.json ./
COPY scripts ./scripts

RUN corepack enable && pnpm install --frozen-lockfile

COPY src ./src
COPY README.md LICENSE ./

RUN pnpm build && pnpm prune --prod

RUN case "${TARGETARCH}" in \
      amd64) yt_dlp_asset="yt-dlp_linux" ;; \
      arm64) yt_dlp_asset="yt-dlp_linux_aarch64" ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
 && curl -fsSL -o "/tmp/${yt_dlp_asset}" "https://github.com/yt-dlp/yt-dlp/releases/download/${YT_DLP_VERSION}/${yt_dlp_asset}" \
 && curl -fsSL -o /tmp/SHA2-256SUMS "https://github.com/yt-dlp/yt-dlp/releases/download/${YT_DLP_VERSION}/SHA2-256SUMS" \
 && (cd /tmp && grep " ${yt_dlp_asset}\$" SHA2-256SUMS | sha256sum -c -) \
 && install -m 0755 "/tmp/${yt_dlp_asset}" /usr/local/bin/yt-dlp \
 && rm -f "/tmp/${yt_dlp_asset}" /tmp/SHA2-256SUMS

FROM node:22-slim AS runtime

ARG SUPERGATEWAY_VERSION=3.4.3

ENV NODE_ENV=production \
    XDG_CACHE_HOME=/opt/media-understanding-cache

WORKDIR /app

RUN npm install -g "supergateway@${SUPERGATEWAY_VERSION}"

COPY --from=builder /usr/local/bin/yt-dlp /usr/local/bin/yt-dlp
COPY --from=builder /opt/media-understanding-cache /opt/media-understanding-cache
COPY --from=builder /app/package.json /app/package.json
COPY --from=builder /app/node_modules /app/node_modules
COPY --from=builder /app/dist /app/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:8000/healthz').then((r) => { if (!r.ok) process.exit(1); }).catch(() => process.exit(1))"

ENTRYPOINT ["supergateway"]
CMD ["--stdio", "node dist/mcp.js", "--outputTransport", "streamableHttp", "--streamableHttpPath", "/mcp", "--stateful", "--port", "8000", "--healthEndpoint", "/healthz"]
