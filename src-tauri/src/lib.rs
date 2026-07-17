use std::env;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::Manager;

pub struct BackendProcess(pub Mutex<Option<Child>>);

#[derive(Serialize)]
struct BackendStatus {
    reachable: bool,
    managed_process: bool,
    exited: bool,
    exit_code: Option<i32>,
    address: String,
}

struct PythonCommand {
    program: PathBuf,
    prefix_args: Vec<String>,
}

fn backend_addr() -> SocketAddr {
    let host = env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".to_string());
    format!("{host}:{port}")
        .parse()
        .expect("valid backend address")
}

fn backend_is_listening() -> bool {
    TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(250)).is_ok()
}

fn backend_is_cephalon() -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(400))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(700)));
    let host = backend_addr();
    let request = format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }
    backend_health_is_compatible(&response)
}

fn backend_health_is_compatible(response: &str) -> bool {
    response.contains("\"service\":\"cephalon\"") && response.contains("\"api_version\":1")
}

#[tauri::command]
fn check_backend() -> bool {
    backend_is_cephalon()
}

fn backend_status(state: &BackendProcess) -> BackendStatus {
    let mut managed_process = false;
    let mut exited = false;
    let mut exit_code = None;
    if let Ok(mut guard) = state.0.lock() {
        if let Some(child) = guard.as_mut() {
            managed_process = true;
            match child.try_wait() {
                Ok(Some(status)) => {
                    exited = true;
                    exit_code = status.code();
                    *guard = None;
                }
                Ok(None) => {}
                Err(_) => {}
            }
        }
    }
    BackendStatus {
        reachable: backend_is_cephalon(),
        managed_process,
        exited,
        exit_code,
        address: backend_addr().to_string(),
    }
}

#[tauri::command]
fn get_backend_status(state: tauri::State<BackendProcess>) -> BackendStatus {
    backend_status(&state)
}

#[tauri::command]
fn minimize_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;
    window.minimize().map_err(|error| error.to_string())
}

#[tauri::command]
fn toggle_maximize_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;
    if window.is_maximized().map_err(|error| error.to_string())? {
        window.unmaximize().map_err(|error| error.to_string())
    } else {
        window.maximize().map_err(|error| error.to_string())
    }
}

#[tauri::command]
fn close_window(app: tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;
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
) {
    let data_dir = default_data_dir();
    let model_dir = default_model_dir(&data_dir);

    command.env("CEPHALON_DATA_DIR", data_dir);
    command.env("CEPHALON_MODEL_DIR", model_dir);
    command.env(
        "CEPHALON_HOST",
        env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
    );
    command.env(
        "CEPHALON_PORT",
        env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".to_string()),
    );
    command.env("CEPHALON_CORS_ORIGINS", "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost");
    command.env("PYTHONNOUSERSITE", "1");

    if let Some(internal) = sidecar_internal {
        let path = prepend_path(env::var("PATH").ok(), &[internal.to_path_buf()]);
        command.env("PATH", path);

        if let Some(root) = repo_root {
            let python_path = prepend_path(
                env::var("PYTHONPATH").ok(),
                &[root.join("python"), internal.to_path_buf()],
            );
            command.env("PYTHONPATH", python_path);
        }
    } else if let Some(root) = repo_root {
        let python_path = prepend_path(env::var("PYTHONPATH").ok(), &[root.join("python")]);
        command.env("PYTHONPATH", python_path);
    }
}

fn python_candidates() -> Vec<PythonCommand> {
    let mut candidates = vec![];
    if cfg!(target_os = "windows") {
        candidates.push(PythonCommand {
            program: PathBuf::from("py"),
            prefix_args: vec!["-3.14".to_string()],
        });
    } else {
        for program in ["python3.14", "python3", "python"] {
            candidates.push(PythonCommand {
                program: PathBuf::from(program),
                prefix_args: vec![],
            });
        }
    }
    candidates
}

fn python_runs(candidate: &PythonCommand) -> bool {
    let mut command = Command::new(&candidate.program);
    for arg in &candidate.prefix_args {
        command.arg(arg);
    }
    command
        .arg("-c")
        .arg("import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn resolve_python_command() -> Option<PythonCommand> {
    for candidate in python_candidates() {
        if python_runs(&candidate) {
            return Some(candidate);
        }
    }
    eprintln!(
        "Cephalon requires Python 3.14. Install it, then ensure `py -3.14` (Windows) or `python3.14` (Linux) is on PATH before rerunning `npm run tauri dev`."
    );
    None
}

fn spawn_dev_backend() -> Option<Child> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent repository")
        .to_path_buf();

    let python = resolve_python_command()?;
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
    apply_backend_env(&mut command, Some(&repo_root), None);

    match command.spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            eprintln!("Failed to start source backend for Tauri dev: {error}");
            None
        }
    }
}

fn spawn_release_backend(app: &tauri::AppHandle) -> Option<Child> {
    let resource_path = app.path().resource_dir().expect("failed to find resources");
    let binary_name = if cfg!(target_os = "windows") {
        "engine.exe"
    } else {
        "engine"
    };
    let binary_path = resource_path
        .join("backend")
        .join("engine")
        .join(binary_name);
    let sidecar_internal = resource_path
        .join("backend")
        .join("engine")
        .join("_internal");

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
        sidecar_internal
            .exists()
            .then_some(sidecar_internal.as_path()),
    );

    match command.spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            eprintln!("Failed to start backend sidecar: {error}");
            None
        }
    }
}

#[tauri::command]
fn restart_backend(
    app: tauri::AppHandle,
    state: tauri::State<BackendProcess>,
) -> Result<BackendStatus, String> {
    if backend_is_cephalon() {
        return Ok(backend_status(&state));
    }
    if backend_is_listening() {
        return Err(format!(
            "Port {} is occupied by a service that is not a compatible Cephalon backend.",
            backend_addr()
        ));
    }
    if env::var("CEPHALON_EXTERNAL_BACKEND").ok().as_deref() == Some("1") {
        return Err("Cephalon is configured to use an external backend. Start that backend, then retry.".to_string());
    }

    let mut guard = state.0.lock().map_err(|_| "Backend process state is unavailable.".to_string())?;
    let already_running = if let Some(child) = guard.as_mut() {
        child.try_wait().map_err(|error| error.to_string())?.is_none()
    } else {
        false
    };
    if already_running {
        drop(guard);
        return Ok(backend_status(&state));
    }
    *guard = if cfg!(debug_assertions) {
        spawn_dev_backend()
    } else {
        spawn_release_backend(&app)
    };
    if guard.is_none() {
        return Err("Cephalon could not start its local backend. Check that the packaged backend and Python runtime are available.".to_string());
    }
    drop(guard);
    Ok(backend_status(&state))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let child = if env::var("CEPHALON_EXTERNAL_BACKEND").ok().as_deref() == Some("1") {
                println!("Cephalon external backend mode enabled; skipping local backend launch.");
                None
            } else if backend_is_cephalon() {
                println!(
                    "Cephalon backend already listening at {}; reusing it.",
                    backend_addr()
                );
                None
            } else if backend_is_listening() {
                return Err(format!(
                    "Port {} is occupied by a service that is not a compatible Cephalon backend.",
                    backend_addr()
                )
                .into());
            } else if cfg!(debug_assertions) {
                spawn_dev_backend()
            } else {
                spawn_release_backend(&app.handle())
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
            get_backend_status,
            restart_backend,
            minimize_window,
            toggle_maximize_window,
            close_window,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::backend_health_is_compatible;

    #[test]
    fn accepts_compatible_cephalon_health_response() {
        let response =
            "HTTP/1.1 200 OK\r\n\r\n{\"service\":\"cephalon\",\"api_version\":1,\"status\":\"ok\"}";
        assert!(backend_health_is_compatible(response));
    }

    #[test]
    fn rejects_unrelated_or_incompatible_services() {
        assert!(!backend_health_is_compatible(
            "HTTP/1.1 200 OK\r\n\r\n{\"status\":\"ok\"}"
        ));
        assert!(!backend_health_is_compatible(
            "{\"service\":\"cephalon\",\"api_version\":2}"
        ));
    }
}
