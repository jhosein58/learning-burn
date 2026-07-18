use clap::Parser;
use std::{
    fs,
    path::PathBuf,
    process::{Command, ExitCode},
};

#[derive(Parser, Debug)]
#[command(version, about = "Run Learning Burn examples")]
struct Args {
    #[arg(short, long)]
    chapter: u8,

    #[arg(short, long)]
    example: u8,

    language: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if let Err(err) = run(args) {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let chapter_dir = PathBuf::from(format!("examples/chapter{:02}", args.chapter));

    if !chapter_dir.exists() {
        return Err(format!("Chapter not found: {}", chapter_dir.display()).into());
    }

    let prefix = format!("e{:02}", args.example);

    let example_dir = fs::read_dir(&chapter_dir)?
        .filter_map(Result::ok)
        .find(|entry| {
            entry.file_type().map(|f| f.is_dir()).unwrap_or(false)
                && entry.file_name().to_string_lossy().starts_with(&prefix)
        })
        .ok_or(format!("Example {} not found", prefix))?
        .path();

    match args.language.as_str() {
        "rust" => {
            let project = example_dir.join("rust");

            if !project.exists() {
                return Err(format!("{} not found", project.display()).into());
            }

            let status = Command::new("cargo")
                .arg("run")
                .current_dir(&project)
                .status()?;

            if !status.success() {
                return Err("Rust example failed".into());
            }
        }

        "python" => {
            let file = example_dir.join("python").join("main.py");

            if !file.exists() {
                return Err(format!("{} not found", file.display()).into());
            }

            let status = Command::new("python3").arg(&file).status()?;

            if !status.success() {
                return Err("Python example failed".into());
            }
        }

        lang => {
            return Err(format!("Unsupported language: {lang}").into());
        }
    }

    Ok(())
}
