fn main() {
    if !std::path::Path::new("backend").exists() {
        std::env::set_var("TAURI_CONFIG", r#"{"bundle":{"resources":[]}}"#);
    }
    tauri_build::build()
}
