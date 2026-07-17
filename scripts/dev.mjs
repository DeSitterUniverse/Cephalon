import net from "node:net";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const backendHost = process.env.CEPHALON_HOST || "127.0.0.1";
const backendPort = Number(process.env.CEPHALON_PORT || "8765");
const viteEntrypoint = resolve(root, "node_modules", "vite", "bin", "vite.js");

function portIsOpen(host, port) {
  return new Promise(resolveOpen => {
    const socket = net.createConnection({ host, port });
    const finish = value => {
      socket.removeAllListeners();
      socket.destroy();
      resolveOpen(value);
    };
    socket.setTimeout(350);
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
    socket.once("timeout", () => finish(false));
  });
}

function pythonCommand() {
  return process.platform === "win32"
    ? { command: "py", args: ["-3.14", resolve(root, "python", "main.py")] }
    : { command: "python3.14", args: [resolve(root, "python", "main.py")] };
}

if (!existsSync(viteEntrypoint)) {
  throw new Error("Vite is not installed. Run npm install before starting Cephalon.");
}

let backend;
if (await portIsOpen(backendHost, backendPort)) {
  console.log(`Reusing backend already listening at http://${backendHost}:${backendPort}.`);
} else {
  const python = pythonCommand();
  console.log(`Starting Cephalon backend at http://${backendHost}:${backendPort}.`);
  backend = spawn(python.command, python.args, {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env, PYTHONNOUSERSITE: "1" },
  });
  backend.once("error", error => console.error(`Could not start the Cephalon backend: ${error.message}`));
}

const frontend = spawn(
  process.execPath,
  [viteEntrypoint, "--host", "127.0.0.1"],
  {
  cwd: root,
  stdio: "inherit",
  env: process.env,
  },
);

function stop(child) {
  if (child && !child.killed) child.kill();
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    stop(frontend);
    stop(backend);
    process.exit(0);
  });
}

frontend.once("exit", code => {
  stop(backend);
  process.exitCode = code ?? 0;
});
frontend.once("error", error => {
  console.error(`Could not start Vite: ${error.message}`);
  stop(backend);
  process.exitCode = 1;
});
