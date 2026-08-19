use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../assets/cephalon.ico");
    println!("cargo:rerun-if-changed=../assets/cephalon.png");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let icon_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/cephalon.ico");
    let icon_path = icon_path
        .to_str()
        .expect("Cephalon icon path must be valid UTF-8");
    let mut resource = winresource::WindowsResource::new();
    resource.set_icon(icon_path);
    resource
        .compile()
        .expect("failed to embed the Cephalon Windows application icon");
}
