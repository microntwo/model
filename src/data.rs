use crate::dataset::{load_image_as_tensor_data, DisasterImageItem};
use burn::{data::dataloader::batcher::Batcher, prelude::*};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            normalizer: Normalizer::<B>::new(&device),
        }
    }
}

impl<B: Backend> Batcher<B, DisasterImageItem, ClassificationBatch<B>>
    for ClassificationBatcher<B>
{
    fn batch(&self, items: Vec<DisasterImageItem>, device: &B::Device) -> ClassificationBatch<B> {
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| load_image_as_tensor_data::<B>(&item.image_path).unwrap())
            .map(|data| {
                Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), &device)
                    .permute([2, 0, 1]) // HWC to CHW
            })
            .map(|tensor| tensor / 255.0)
            .collect();

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    &device,
                )
            })
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        let images = self.normalizer.to_device(&device).normalize(images);

        ClassificationBatch { images, targets }
    }
}
