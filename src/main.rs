#![recursion_limit = "256"]

use anyhow::Result;
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::optim::AdamWConfig;
use clap::{Parser, Subcommand};
use model::{infer, train, training::TrainingConfig};
use std::path::PathBuf;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train {
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
    },
    Infer {
        #[arg(long)]
        image_path: PathBuf,
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
    },
}

fn main() -> Result<()> {
    let file_appender = tracing_appender::rolling::daily("logs", "training.log");
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = fmt::layer()
        .with_writer(non_blocking_writer)
        .with_ansi(false)
        .with_filter(EnvFilter::new("info"));

    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(EnvFilter::new("warn"));

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .init();

    let device = WgpuDevice::default();
    tracing::info!("Using device: {:?}", device);

    let cli = Cli::parse();

    match cli.command {
        Commands::Train { artifact_dir } => {
            tracing::info!("Starting training mode with Hybrid CNN v1.0.0...");
            let config = TrainingConfig::new(AdamWConfig::new())
                .with_batch_size(32)
                .with_num_epochs(5)
                .with_learning_rate(5e-5);

            train::<MyAutodiffBackend>(device, &artifact_dir, config);
        }
        Commands::Infer {
            image_path,
            artifact_dir,
        } => {
            tracing::info!("Starting inference mode with Hybrid CNN v1.0.0...");
            if !image_path.exists() {
                anyhow::bail!("Image file not found: {}", image_path.display());
            }
            infer::<MyBackend>(device, &artifact_dir, &image_path)?;
        }
    }

    Ok(())
}
