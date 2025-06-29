use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rand::Rng;

fn apply_cutout(img: &mut DynamicImage, rng: &mut impl Rng) {
    let (width, height) = img.dimensions();
    let cutout_width = rng.random_range((width / 8)..(width / 4));
    let cutout_height = rng.random_range((height / 8)..(height / 4));
    let x_offset = rng.random_range(0..=(width - cutout_width));
    let y_offset = rng.random_range(0..=(height - cutout_height));
    let cutout_box = image::DynamicImage::new_rgba8(cutout_width, cutout_height);
    img.copy_from(&cutout_box, x_offset, y_offset).ok();
}

pub fn apply_augmentations(img: DynamicImage, is_train: bool) -> DynamicImage {
    if !is_train {
        return img;
    }

    let mut rng = rand::rng();
    let mut img = img;

    if rng.random_bool(0.5) {
        img = img.fliph();
    }

    if rng.random_bool(0.5) {
        let angle_deg: f32 = rng.random_range(-15.0..15.0);
        let angle_rad = angle_deg.to_radians();
        let rgba_img = img.to_rgba8();

        let rotated = rotate_about_center(
            &rgba_img,
            angle_rad,
            Interpolation::Bilinear,
            Rgba([0, 0, 0, 0]),
        );
        img = DynamicImage::ImageRgba8(rotated);
    }

    if rng.random_bool(0.3) {
        let brightness_factor: f32 = rng.random_range(-0.25..0.25);
        img = adjust_brightness(img, brightness_factor);
    }

    if rng.random_bool(0.3) {
        let contrast_factor: f32 = rng.random_range(-0.2..0.4);
        img = img.adjust_contrast(contrast_factor);
    }

    if rng.random_bool(0.2) {
        let sigma: f32 = rng.random_range(0.1..1.5);
        img = img.blur(sigma);
    }

    if rng.random_bool(0.5) {
        apply_cutout(&mut img, &mut rng);
    }

    img
}

fn adjust_brightness(img: DynamicImage, factor: f32) -> DynamicImage {
    let mut rgb_img = img.to_rgb8();
    for pixel in rgb_img.pixels_mut() {
        let r = (pixel[0] as f32 * (1.0 + factor)).clamp(0.0, 255.0) as u8;
        let g = (pixel[1] as f32 * (1.0 + factor)).clamp(0.0, 255.0) as u8;
        let b = (pixel[2] as f32 * (1.0 + factor)).clamp(0.0, 255.0) as u8;
        *pixel = image::Rgb([r, g, b]);
    }
    DynamicImage::ImageRgb8(rgb_img)
}
