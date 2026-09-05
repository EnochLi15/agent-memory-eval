FROM node:24.18.0-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build && npm prune --omit=dev
FROM node:24.18.0-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY python/requirements.lock /app/python/requirements.lock
RUN python3 -m venv /opt/eval-python && /opt/eval-python/bin/pip install --no-cache-dir -r /app/python/requirements.lock
ENV EVAL_PYTHON=/opt/eval-python/bin/python
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json ./
COPY contracts ./contracts
COPY python ./python
COPY configs ./configs
COPY scripts ./scripts
RUN mkdir -p /app/artifacts /app/.data && chown node:node /app/artifacts /app/.data
USER node
ENTRYPOINT ["node","dist/cli.js"]
