use std::env;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::Manager;

pub struct BackendProcess(pub Mutex<Option<Child>>);

fn backend_addr() -> SocketAddr {
    "127.0.0.1:8765".parse().expect("valid backend address")
}

fn backend_is_listening() -> bool {
    TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(250)).is_ok()
}

#[tauri::command]
fn check_backend() -> bool {
    backend_is_listening()
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

fn apply_backend_env(command: &mut Command, repo_root: Option<&Path>, sidecar_internal: Option<&Path>) {
    let data_dir = default_data_dir();
    let model_dir = default_model_dir(&data_dir);

    command.env("CEPHALON_DATA_DIR", data_dir);
    command.env("CEPHALON_MODEL_DIR", model_dir);
    command.env("CEPHALON_CORS_ORIGINS", "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost");
    command.env("CEPHALON_REQUIRE_VULKAN", "1");
    command.env("CEPHALON_LLAMA_VERBOSE", "0");

    if let Some(internal) = sidecar_internal {
        let dll_dir = internal.join("llama_cpp").join("lib");
        command.env("CEPHALON_LLAMA_DLL_DIR", &dll_dir);
        command.env("LLAMA_CPP_LIB_PATH", &dll_dir);
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

fn spawn_dev_backend() -> Option<Child> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent repository")
        .to_path_buf();

    let python = if cfg!(target_os = "windows") {
        repo_root.join(".venv").join("Scripts").join("python.exe")
    } else {
        repo_root.join(".venv").join("bin").join("python")
    };
    let python = if python.exists() { python } else { PathBuf::from("python") };
    let sidecar_internal = repo_root.join("src-tauri").join("backend").join("engine").join("_internal");

    let mut command = Command::new(python);
    command
        .arg(repo_root.join("python").join("main.py"))
        .current_dir(&repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_backend_env(
        &mut command,
        Some(&repo_root),
        sidecar_internal.exists().then_some(sidecar_internal.as_path()),
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
    apply_backend_env(&mut command, None, sidecar_internal.exists().then_some(sidecar_internal.as_path()));

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
            let child = if backend_is_listening() {
                println!("Cephalon backend already listening on 127.0.0.1:8765; reusing it.");
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
        .invoke_handler(tauri::generate_handler![check_backend])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
