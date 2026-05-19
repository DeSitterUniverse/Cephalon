use std::env;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::Manager;

pub struct BackendProcess(pub Mutex<Option<Child>>);

struct PythonCommand {
    program: PathBuf,
    prefix_args: Vec<String>,
}

fn backend_addr() -> SocketAddr {
    let host = env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".to_string());
    format!("{host}:{port}").parse().expect("valid backend address")
}

fn backend_is_listening() -> bool {
    TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(250)).is_ok()
}

#[tauri::command]
fn check_backend() -> bool {
    backend_is_listening()
}

#[tauri::command]
fn minimize_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app.get_webview_window("main").ok_or_else(|| "main window not found".to_string())?;
    window.minimize().map_err(|error| error.to_string())
}

#[tauri::command]
fn toggle_maximize_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app.get_webview_window("main").ok_or_else(|| "main window not found".to_string())?;
    if window.is_maximized().map_err(|error| error.to_string())? {
        window.unmaximize().map_err(|error| error.to_string())
    } else {
        window.maximize().map_err(|error| error.to_string())
    }
}

#[tauri::command]
fn close_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app.get_webview_window("main").ok_or_else(|| "main window not found".to_string())?;
    window.close().map_err(|error| error.to_string())
}

fn home_dir() -> PathBuf {
    env::var_os("USERPROFILE")
        .or_else(|| env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_data_dir() -> PathBuf {
    env::var_os("CEPHALON_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join("cephalon-data"))
}

fn default_model_dir(data_dir: &Path) -> PathBuf {
    env::var_os("CEPHALON_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("models"))
}

fn prepend_path(existing: Option<String>, paths: &[PathBuf]) -> String {
    let mut parts: Vec<PathBuf> = paths.iter().filter(|path| path.exists()).cloned().collect();
    if let Some(existing) = existing {
        parts.extend(env::split_paths(&existing));
    }
    env::join_paths(parts)
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn apply_backend_env(
    command: &mut Command,
    repo_root: Option<&Path>,
    sidecar_internal: Option<&Path>,
    dev_llama_lib: Option<&Path>,
) {
    let data_dir = default_data_dir();
    let model_dir = default_model_dir(&data_dir);

    command.env("CEPHALON_DATA_DIR", data_dir);
    command.env("CEPHALON_MODEL_DIR", model_dir);
    command.env("CEPHALON_HOST", env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()));
    command.env("CEPHALON_PORT", env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".to_string()));
    command.env("CEPHALON_CORS_ORIGINS", "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost");
    command.env("CEPHALON_LLAMA_VERBOSE", "0");
    command.env("PYTHONNOUSERSITE", "1");

    if let Some(llama_lib) = dev_llama_lib {
        command.env("CEPHALON_LLAMA_DLL_DIR", llama_lib);
        command.env("LLAMA_CPP_LIB_PATH", llama_lib);
        let path = prepend_path(env::var("PATH").ok(), &[llama_lib.to_path_buf()]);
        command.env("PATH", path);
    }

    if let Some(internal) = sidecar_internal {
        let dll_dir = internal.join("llama_cpp").join("lib");
        let path = prepend_path(env::var("PATH").ok(), &[dll_dir, internal.to_path_buf()]);
        command.env("PATH", path);

        if let Some(root) = repo_root {
            let python_path = prepend_path(env::var("PYTHONPATH").ok(), &[root.join("python"), internal.to_path_buf()]);
            command.env("PYTHONPATH", python_path);
        }
    } else if let Some(root) = repo_root {
        let python_path = prepend_path(env::var("PYTHONPATH").ok(), &[root.join("python")]);
        command.env("PYTHONPATH", python_path);
    }
}

fn python_candidates() -> Vec<PythonCommand> {
    let mut candidates = vec![PythonCommand {
        program: PathBuf::from("python"),
        prefix_args: vec![],
    }];
    if cfg!(target_os = "windows") {
        candidates.push(PythonCommand {
            program: PathBuf::from("py"),
            prefix_args: vec!["-3".to_string()],
        });
    }
    candidates
}

fn python_runs(candidate: &PythonCommand) -> bool {
    let mut command = Command::new(&candidate.program);
    for arg in &candidate.prefix_args {
        command.arg(arg);
    }
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.status().map(|status| status.success()).unwrap_or(false)
}

fn resolve_python_command() -> Option<PythonCommand> {
    for candidate in python_candidates() {
        if python_runs(&candidate) {
            return Some(candidate);
        }
    }
    eprintln!(
        "Cephalon could not find Python on PATH. Enable the Windows Python app execution alias or add python.exe/py.exe to PATH, then rerun npm run tauri dev."
    );
    None
}

fn discover_dev_llama_lib(python: &PythonCommand) -> Option<PathBuf> {
    let mut command = Command::new(&python.program);
    for arg in &python.prefix_args {
        command.arg(arg);
    }
    command
        .arg("-c")
        .arg("import pathlib, llama_cpp; p = pathlib.Path(llama_cpp.__file__).resolve().parent / 'lib'; v = p / 'ggml-vulkan.dll'; print(p if v.exists() else '')")
        .stdin(Stdio::null());
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Failed to inspect local Python llama_cpp package: {error}");
            return None;
        }
    };
    if !output.status.success() {
        eprintln!(
            "Local Python can start, but llama_cpp is not importable. Install the Vulkan-enabled llama-cpp-python wheel before running Tauri dev."
        );
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        eprintln!(
            "Local Python has llama_cpp, but ggml-vulkan.dll was not found. Rebuild llama-cpp-python with CMAKE_ARGS=-DGGML_VULKAN=on."
        );
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn spawn_dev_backend() -> Option<Child> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent repository")
        .to_path_buf();

    let python = resolve_python_command()?;
    let dev_llama_lib = discover_dev_llama_lib(&python)?;
    let mut command = Command::new(&python.program);
    for arg in &python.prefix_args {
        command.arg(arg);
    }
    command
        .arg(repo_root.join("python").join("main.py"))
        .current_dir(&repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_backend_env(
        &mut command,
        Some(&repo_root),
        None,
        Some(dev_llama_lib.as_path()),
    );

    match command.spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            eprintln!("Failed to start source backend for Tauri dev: {error}");
            None
        }
    }
}

fn spawn_release_backend(app: &tauri::App) -> Option<Child> {
    let resource_path = app.path().resource_dir().expect("failed to find resources");
    let binary_name = if cfg!(target_os = "windows") { "engine.exe" } else { "engine" };
    let binary_path = resource_path.join("backend").join("engine").join(binary_name);
    let sidecar_internal = resource_path.join("backend").join("engine").join("_internal");

    if !binary_path.exists() {
        eprintln!("Backend sidecar not found at {}.", binary_path.display());
        return None;
    }

    let mut command = Command::new(binary_path);
    command
        .current_dir(resource_path)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_backend_env(
        &mut command,
        None,
        sidecar_internal.exists().then_some(sidecar_internal.as_path()),
        None,
    );

    match command.spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            eprintln!("Failed to start backend sidecar: {error}");
            None
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let child = if env::var("CEPHALON_EXTERNAL_BACKEND").ok().as_deref() == Some("1") {
                println!("Cephalon external backend mode enabled; skipping local backend launch.");
                None
            } else if backend_is_listening() {
                println!("Cephalon backend already listening at {}; reusing it.", backend_addr());
                None
            } else if cfg!(debug_assertions) {
                spawn_dev_backend()
            } else {
                spawn_release_backend(app)
            };
            app.manage(BackendProcess(Mutex::new(child)));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.app_handle().state::<BackendProcess>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                    }
                };
            }
        })
        .invoke_handler(tauri::generate_handler![
            check_backend,
            minimize_window,
            toggle_maximize_window,
            close_window,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
