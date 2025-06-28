use crate::{
    data::ClassificationBatcher, dataset::DisasterImageItem, model::Cnn, training::NUM_CLASSES,
};
use anyhow::Result;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use std::path::Path;

pub fn infer<B: Backend>(device: B::Device, artifact_dir: &str, image_path: &Path) -> Result<()> {
    tracing::info!("Loading model from: {}", artifact_dir);

    let _config = crate::training::TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
    let record = CompactRecorder::new().load(format!("{artifact_dir}/model").into(), &device)?;

    let model: Cnn<B> = Cnn::new(NUM_CLASSES, &device).load_record(record);

    tracing::info!("Loading and preparing image: {:?}", image_path);
    let batcher = ClassificationBatcher::<B>::new(device.clone());

    let item = DisasterImageItem {
        image_path: image_path.to_path_buf(),
        label: 0,
    };

    let batch = batcher.batch(vec![item], &device);

    tracing::info!("Running inference...");
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_data();
    let predicted_idx = predicted.iter::<i64>().next().unwrap() as usize;

    let class_names = ["fire_and_smoke", "flood", "wildfire", "normal"];
    let predicted_class = class_names.get(predicted_idx).unwrap_or(&"unknown");

    println!("\nPrediction for '{}':", image_path.display());
    println!("  - Class Index: {}", predicted_idx);
    println!("  - Class Name:  {}", predicted_class);

    Ok(())
}
