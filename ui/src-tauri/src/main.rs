// src-tauri/src/main.rs
// Tauri shell — spawns Python backend, bridges JSON lines, forwards to frontend.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Manager, State};

struct PythonProcess {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

type ProcState = Arc<Mutex<PythonProcess>>;

fn spawn_python(app: AppHandle, proc_state: ProcState) {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python".to_string());
    let script = std::env::var("AGENT_SCRIPT").unwrap_or_else(|_| "./src/api.py".to_string());

    let mut child = Command::new(&python)
        .arg(&script)
        .arg("run")
        .arg("agent ready")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn Python backend");

    let stdout = child.stdout.take().expect("no stdout");
    let stdin  = child.stdin.take().expect("no stdin");

    // store stdin handle for sending commands
    {
        let mut p = proc_state.lock().unwrap();
        p.stdin = Some(stdin);
        p.child = Some(child);
    }

    // thread: read Python stdout → emit to frontend
    let app_clone = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app_clone.emit_all("agent-message", json);
                }
            }
        }
    });
}

#[tauri::command]
fn send_cmd(
    cmd: String,
    extra: Option<String>,
    proc: State<ProcState>,
) -> Result<(), String> {
    let mut p = proc.lock().map_err(|e| e.to_string())?;
    if let Some(stdin) = p.stdin.as_mut() {
        let mut msg = serde_json::json!({ "cmd": cmd });
        if let Some(e) = extra {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&e) {
                if let serde_json::Value::Object(map) = parsed {
                    if let serde_json::Value::Object(ref mut m) = msg {
                        m.extend(map);
                    }
                }
            }
        }
        let line = serde_json::to_string(&msg).map_err(|e| e.to_string())? + "\n";
        stdin.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    let proc_state: ProcState = Arc::new(Mutex::new(PythonProcess {
        child: None,
        stdin: None,
    }));

    tauri::Builder::default()
        .manage(proc_state.clone())
        .invoke_handler(tauri::generate_handler![send_cmd])
        .setup(move |app| {
            let handle = app.handle();
            let ps = proc_state.clone();
            // spawn Python after window is ready
            thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(500));
                spawn_python(handle, ps);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Tauri app");
}
