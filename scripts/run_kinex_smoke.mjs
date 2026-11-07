#!/usr/bin/env node
// Bounded action acceptance smoke without an LLM provider.

import { copyFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [
  workspace = process.env.KINEX_WORKSPACE,
  outputDirectory = path.join(
    workspace ?? process.cwd(),
    "results",
    `smoke-${new Date().toISOString().replaceAll(":", "-")}`,
  ),
  coreDist = process.env.KINEX_CORE_DIST,
] = process.argv.slice(2);

if (!workspace || !coreDist) {
  throw new Error(
    "usage: run_kinex_smoke.mjs <workspace> <output-dir> <kinex-core-dist>",
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
  sessionID: "tableclear-device-smoke",
  signal: AbortSignal.timeout(900_000),
};
const native = Boolean(process.env.TABLECLEAR_NATIVE_OSC_SOCKET);

try {
  const devices = await runtime.devices.list();
  const deviceNames = devices.map((device) => device.name);
  for (const device of devices) {
    await (await runtime.devices.load(device.name)).ready;
  }

  const call = await runtime.tools.call(
    "run_vla_subtask",
    {
      instruction: native
        ? "put both the alphabet soup and the tomato sauce in the basket"
        : "pick up the object on the table and put it in the basket",
      max_actions: native ? 10 : 1,
      action_hz: native ? 20 : 30,
    },
    invocation,
  );
  if (call.isError) {
    throw new Error(call.output);
  }

  const subtask = JSON.parse(call.output);
  if (native && subtask.control_protocol !== "libero_native_osc") {
    throw new Error("Kinex did not use the native OSC session");
  }
  await mkdir(outputDirectory, { recursive: true });
  subtask.before_images = await persistImages(
    subtask.before_images,
    outputDirectory,
    "before",
  );
  subtask.after_images = await persistImages(
    subtask.after_images,
    outputDirectory,
    "after",
  );
  const result = {
    passed: true,
    devices: deviceNames,
    subtask,
  };
  await writeFile(
    path.join(outputDirectory, "device-smoke.json"),
    `${JSON.stringify(result, null, 2)}\n`,
  );
  console.log(JSON.stringify(result, null, 2));
} finally {
  await runtime.dispose();
}

async function persistImages(paths, outputDirectory, phase) {
  if (!Array.isArray(paths)) {
    return paths;
  }
  const output = [];
  for (const [index, source] of paths.entries()) {
    const destination = path.join(
      outputDirectory,
      `${phase}-${index === 0 ? "head" : "wrist"}.png`,
    );
    await copyFile(source, destination);
    output.push(destination);
  }
  return output;
}
