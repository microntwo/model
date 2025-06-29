use crate::{
    data::{ClassificationBatch, ClassificationBatcher},
    dataset::{DisasterImageDataset, DisasterLoader},
    model::Cnn,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};
use std::time::Instant;

pub const NUM_CLASSES: usize = 5;

impl<B: Backend> Cnn<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(device: B::Device, artifact_dir: &str, config: TrainingConfig) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DisasterImageDataset::disaster_train().unwrap());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(DisasterImageDataset::disaster_valid().unwrap());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            Cnn::new(NUM_CLASSES, &device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    let elapsed = now.elapsed().as_secs();
    tracing::info!("Total training time: {}m {}s", (elapsed / 60), elapsed % 60);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
