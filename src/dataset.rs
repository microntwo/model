use crate::augmentation::apply_augmentations;
use anyhow::Result;
use burn::data::dataset::Dataset;
use burn::prelude::{Backend, Shape, TensorData};
use image::{imageops::FilterType, GenericImageView};
use std::path::{Path, PathBuf};

pub const IMAGE_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub struct DisasterImageItem {
    pub image_path: PathBuf,
    pub label: usize,
    pub is_train: bool,
}

#[derive(Debug, Clone)]
pub struct DisasterImageDataset {
    items: Vec<DisasterImageItem>,
}

impl DisasterImageDataset {
    pub fn new_classification(split: &str, is_train: bool) -> Result<Self> {
        let root = Path::new("DisasterData_split").join(split);
        let mut class_dirs = std::fs::read_dir(&root)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();

        class_dirs.sort();

        let mut items = Vec::new();
        for (label, class_dir) in class_dirs.iter().enumerate() {
            for entry in std::fs::read_dir(class_dir)? {
                let path = entry?.path();
                if path.is_file() {
                    items.push(DisasterImageItem {
                        image_path: path,
                        label,
                        is_train,
                    });
                }
            }
        }
        Ok(Self { items })
    }
}

impl Dataset<DisasterImageItem> for DisasterImageDataset {
    fn get(&self, index: usize) -> Option<DisasterImageItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

pub trait DisasterLoader {
    fn disaster_train() -> Result<Self>
    where
        Self: Sized;
    fn disaster_valid() -> Result<Self>
    where
        Self: Sized;
    fn disaster_test() -> Result<Self>
    where
        Self: Sized;
}

impl DisasterLoader for DisasterImageDataset {
    fn disaster_train() -> Result<Self> {
        Self::new_classification("train", true)
    }
    fn disaster_valid() -> Result<Self> {
        Self::new_classification("valid", false)
    }
    fn disaster_test() -> Result<Self> {
        Self::new_classification("test", false)
    }
}

pub fn load_image_as_tensor_data<B: Backend>(path: &Path, is_train: bool) -> Result<TensorData> {
    let image = image::ImageReader::open(path)?.decode()?;
    let image = apply_augmentations(image, is_train);
    let resized = image.resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::Triangle);
    let (width, height) = resized.dimensions();

    let image_bytes = resized.to_rgb8().to_vec();

    let data = TensorData::new(
        image_bytes,
        Shape::new([height as usize, width as usize, 3]),
    );
    Ok(data)
}
