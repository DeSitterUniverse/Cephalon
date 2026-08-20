use std::env;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub reachable: bool,
    pub managed_process: bool,
    pub exited: bool,
    pub exit_code: Option<i32>,
    pub address: String,
}

pub struct BackendService {
    child: Mutex<Option<Child>>,
}

impl BackendService {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
        }
    }

    pub fn start(&self) -> Result<BackendStatus, String> {
        if backend_is_cephalon() {
            return Ok(self.status());
        }
        if backend_is_listening() {
            return Err(format!(
                "Port {} is occupied by a service that is not a compatible Cephalon backend.",
                backend_addr()
            ));
        }
        if env::var("CEPHALON_EXTERNAL_BACKEND").ok().as_deref() == Some("1") {
            return Err(
                "Cephalon is configured to use an external backend. Start that backend, then retry."
                    .to_string(),
            );
        }

        let mut guard = self
            .child
            .lock()
            .map_err(|_| "Backend process state is unavailable.".to_string())?;
        let already_running = if let Some(child) = guard.as_mut() {
            child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
        } else {
            false
        };
        if already_running {
            drop(guard);
            return Ok(self.status());
        }

        let child = if cfg!(debug_assertions) {
            spawn_dev_backend()
        } else {
            spawn_release_backend()
        };
        if child.is_none() {
            return Err(
                "Cephalon could not start its local backend. Check that the packaged backend and Python runtime are available."
                    .to_string(),
            );
        }
        *guard = child;
        drop(guard);
        Ok(self.status())
    }

    #[allow(dead_code)]
    pub fn restart(&self) -> Result<BackendStatus, String> {
        self.shutdown();
        self.start()
    }

    pub fn status(&self) -> BackendStatus {
        let mut managed_process = false;
        let mut exited = false;
        let mut exit_code = None;
        if let Ok(mut guard) = self.child.lock() {
            if let Some(child) = guard.as_mut() {
                managed_process = true;
                match child.try_wait() {
                    Ok(Some(status)) => {
                        exited = true;
                        exit_code = status.code();
                        *guard = None;
                    }
                    Ok(None) | Err(_) => {}
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

    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                terminate_process_tree(&mut child);
            }
        }
    }
}

impl Default for BackendService {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BackendService {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn backend_addr() -> SocketAddr {
    let host = env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".to_string());
    format!("{host}:{port}")
        .parse()
        .expect("CEPHALON_HOST and CEPHALON_PORT must form a valid socket address")
}

fn backend_is_listening() -> bool {
    TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(250)).is_ok()
}

fn backend_is_cephalon() -> bool {
    if let Some(response) = backend_probe("/identity", Duration::from_millis(750)) {
        if backend_identity_is_compatible(&response) {
            return true;
        }
    }

    // Keep the health fallback for older external or packaged backends that do
    // not yet expose `/identity`. Health remains the readiness endpoint, not the
    // normal process-identity path.
    backend_probe("/health", Duration::from_secs(4))
        .is_some_and(|response| backend_health_is_compatible(&response))
}

fn backend_probe(path: &str, timeout: Duration) -> Option<String> {
    let mut stream =
        TcpStream::connect_timeout(&backend_addr(), Duration::from_millis(400)).ok()?;
    let _ = stream.set_read_timeout(Some(timeout));
    let host = backend_addr();
    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response)
}

fn backend_identity_is_compatible(response: &str) -> bool {
    response.contains("\"service\":\"cephalon\"") && response.contains("\"api_version\":1")
}

fn backend_health_is_compatible(response: &str) -> bool {
    response.contains("\"service\":\"cephalon\"") && response.contains("\"api_version\":1")
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
    command.env("PYTHONNOUSERSITE", "1");

    if let Some(internal) = sidecar_internal {
        command.env(
            "PATH",
            prepend_path(env::var("PATH").ok(), &[internal.to_path_buf()]),
        );
        if let Some(root) = repo_root {
            command.env(
                "PYTHONPATH",
                prepend_path(
                    env::var("PYTHONPATH").ok(),
                    &[root.join("python"), internal.to_path_buf()],
                ),
            );
        }
    } else if let Some(root) = repo_root {
        command.env(
            "PYTHONPATH",
            prepend_path(env::var("PYTHONPATH").ok(), &[root.join("python")]),
        );
    }
}

struct PythonCommand {
    program: PathBuf,
    prefix_args: Vec<String>,
}

fn python_candidates() -> Vec<PythonCommand> {
    if cfg!(target_os = "windows") {
        vec![PythonCommand {
            program: PathBuf::from("py"),
            prefix_args: vec!["-3.14".to_string()],
        }]
    } else {
        ["python3.14", "python3", "python"]
            .into_iter()
            .map(|program| PythonCommand {
                program: PathBuf::from(program),
                prefix_args: vec![],
            })
            .collect()
    }
}

fn resolve_python_command() -> Option<PythonCommand> {
    for candidate in python_candidates() {
        let mut command = Command::new(&candidate.program);
        for arg in &candidate.prefix_args {
            command.arg(arg);
        }
        command
            .arg("-c")
            .arg("import sys\nif sys.version_info[:2] == (3, 14): print(sys.executable)\nelse: raise SystemExit(1)")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Ok(output) = command.output() {
            if output.status.success() {
                if cfg!(target_os = "windows") {
                    let executable = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !executable.is_empty() {
                        return Some(PythonCommand {
                            program: PathBuf::from(executable),
                            prefix_args: Vec::new(),
                        });
                    }
                } else {
                    return Some(candidate);
                }
            }
        }
    }
    eprintln!(
        "Cephalon requires Python 3.14. Install it, then ensure `py -3.14` (Windows) or `python3.14` (Linux) is on PATH."
    );
    None
}

fn spawn_dev_backend() -> Option<Child> {
    let repo_root = dev_repo_root();
    let python = resolve_python_command()?;
    let script = repo_root.join("python").join("main.py");
    if !script.exists() {
        eprintln!(
            "Source backend entrypoint not found at {}.",
            script.display()
        );
        return None;
    }
    let mut command = Command::new(&python.program);
    for arg in &python.prefix_args {
        command.arg(arg);
    }
    command
        .arg(script)
        .current_dir(&repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_backend_env(&mut command, Some(&repo_root), None);
    configure_process_group(&mut command);
    command.spawn().map_or_else(
        |error| {
            eprintln!("Failed to start source backend: {error}");
            None
        },
        Some,
    )
}

fn dev_repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

fn spawn_release_backend() -> Option<Child> {
    let binary_name = if cfg!(target_os = "windows") {
        "engine.exe"
    } else {
        "engine"
    };
    let resource_root = env::var_os("CEPHALON_RESOURCE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::current_exe()
                .ok()?
                .parent()
                .map(Path::to_path_buf)
        })?;
    let candidates = [
        resource_root.join("backend").join("engine"),
        resource_root
            .join("resources")
            .join("backend")
            .join("engine"),
    ];
    let engine_dir = candidates.iter().find(|path| path.exists())?.to_path_buf();
    let binary_path = engine_dir.join(binary_name);
    let sidecar_internal = engine_dir.join("_internal");
    if !binary_path.exists() {
        eprintln!("Backend sidecar not found at {}.", binary_path.display());
        return None;
    }
    let mut command = Command::new(binary_path);
    command
        .current_dir(&resource_root)
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
    configure_process_group(&mut command);
    command.spawn().map_or_else(
        |error| {
            eprintln!("Failed to start backend sidecar: {error}");
            None
        },
        Some,
    )
}

fn terminate_process_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .status();
        let _ = child.wait();
    }

    #[cfg(unix)]
    {
        let process_group = -(child.id() as i32);
        unsafe {
            libc::kill(process_group, libc::SIGTERM);
        }
        for _ in 0..20 {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
        unsafe {
            libc::kill(process_group, libc::SIGKILL);
        }
        let _ = child.wait();
    }
}

fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
    }

    #[cfg(windows)]
    let _ = command;
}

#[cfg(test)]
mod tests {
    use super::{backend_health_is_compatible, backend_identity_is_compatible, dev_repo_root};

    #[test]
    fn development_backend_uses_workspace_python_entrypoint() {
        let root = dev_repo_root();
        assert!(root.join("python").join("main.py").is_file());
        assert!(!root.join("native").join("python").join("main.py").exists());
    }

    #[test]
    fn accepts_cheap_compatible_cephalon_identity_response() {
        let response = "HTTP/1.1 200 OK\r\n\r\n{\"service\":\"cephalon\",\"api_version\":1}";
        assert!(backend_identity_is_compatible(response));
    }

    #[test]
    fn rejects_incompatible_identity_versions() {
        assert!(!backend_identity_is_compatible(
            "HTTP/1.1 200 OK\r\n\r\n{\"service\":\"cephalon\",\"api_version\":2}"
        ));
    }

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
