#!/usr/bin/env node
// Direct-VLA smoke episode without an LLM provider.

import { randomUUID } from "node:crypto";
import { copyFile, mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [
  workspace = process.env.KINEX_WORKSPACE,
  outputDirectory,
  coreDist = process.env.KINEX_CORE_DIST,
  instruction,
  maxChunksRaw = "20",
  actionsPerChunkRaw = "10",
] = process.argv.slice(2);
const maxChunks = Number(maxChunksRaw);
const actionsPerChunk = Number(actionsPerChunkRaw);

if (
  !workspace ||
  !outputDirectory ||
  !coreDist ||
  !instruction ||
  !Number.isInteger(maxChunks) ||
  maxChunks < 1 ||
  !Number.isInteger(actionsPerChunk) ||
  !Number.isInteger(actionsPerChunk) ||
  actionsPerChunk < 1 ||
  actionsPerChunk > 50
) {
  throw new Error(
    "usage: run_direct_episode.mjs <workspace> <output-dir> " +
      "<kinex-core-dist> <instruction> [max-chunks] [actions-per-chunk]",
  );
}

const { createKinexRuntime } = await import(
  pathToFileURL(path.join(coreDist, "index.js")).href
);
const runtime = createKinexRuntime({
  directory: path.resolve(workspace),
  host: {
    ask: async () => {},
    questions: async () => ({}),
  },
});
const invocation = {
  sessionID: "tableclear-direct-episode",
  signal: AbortSignal.timeout(1_800_000),
};
const endpoint = process.env.ROBOENV_ENDPOINT ?? "http://127.0.0.1:8123";
const record = {
  schema: "tableclear/direct-episode-smoke/1",
  started_at: new Date().toISOString(),
  instruction,
  max_chunks: maxChunks,
  actions_per_chunk: actionsPerChunk,
  chunks: [],
  status: "running",
};

try {
  await mkdir(outputDirectory, { recursive: true });
  const devices = await runtime.devices.list();
  for (const device of devices) {
    await (await runtime.devices.load(device.name)).ready;
  }
  record.devices = devices.map((device) => device.name);
  await writeRecord();

  for (let index = 0; index < maxChunks; index += 1) {
    try {
      const call = await runtime.tools.call(
        "run_vla_subtask",
        {
          instruction,
          max_actions: actionsPerChunk,
          action_hz: 30,
        },
        invocation,
      );
      if (call.isError) {
        throw new Error(call.output);
      }
      const subtask = JSON.parse(call.output);
      subtask.before_images = await persistImages(
        subtask.before_images,
        outputDirectory,
        index,
        "before",
      );
      subtask.after_images = await persistImages(
        subtask.after_images,
        outputDirectory,
        index,
        "after",
      );
      record.chunks.push({ index, subtask });
    } catch (error) {
      record.chunks.push({
        index,
        error: error instanceof Error ? error.message : String(error),
      });
      record.status = "interrupted";
      await writeRecord();
      break;
    }

    const status = await publicStatus(endpoint);
    record.chunks.at(-1).simulation = status.simulation;
    await writeRecord();
    console.error(
      JSON.stringify({
        chunk: index,
        submitted: record.chunks.at(-1).subtask.actions_submitted,
        terminal: status.simulation?.terminal,
      }),
    );
    if (status.simulation?.terminal) {
      record.status = "terminal";
      break;
    }
  }

  if (record.status === "running") {
    record.status = "chunk_budget_exhausted";
  }
} finally {
  record.finished_at = new Date().toISOString();
  await writeRecord();
  await runtime.dispose();
}

console.log(JSON.stringify(record, null, 2));

async function persistImages(paths, directory, chunk, phase) {
  if (!Array.isArray(paths)) {
    return paths;
  }
  const output = [];
  for (const [cameraIndex, source] of paths.entries()) {
    const destination = path.join(
      directory,
      `${String(chunk).padStart(3, "0")}-${phase}-` +
        `${cameraIndex === 0 ? "head" : "wrist"}.png`,
    );
    await copyFile(source, destination);
    output.push(destination);
  }
  return output;
}

async function writeRecord() {
  const destination = path.join(outputDirectory, "episode.json");
  const temporary = `${destination}.tmp`;
  await writeFile(temporary, `${JSON.stringify(record, null, 2)}\n`);
  await rename(temporary, destination);
}

async function publicStatus(serviceEndpoint) {
  const identity = await fetch(serviceEndpoint, {
    method: "GET",
    signal: AbortSignal.timeout(10_000),
  }).then((response) => response.json());
  const request = {
    protocol: "roboenv/1",
    session: identity.session,
    generation: identity.generation,
    request_id: `tableclear-${randomUUID()}`,
    deadline_s: 10,
    op: "simulation.status",
    args: {},
  };
  const response = await fetch(serviceEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: `${JSON.stringify(request)}\n`,
    signal: AbortSignal.timeout(15_000),
  }).then((value) => value.json());
  if (!response.ok) {
    throw new Error(JSON.stringify(response.error ?? response));
  }
  return response.result;
}
