use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct SeparableConv2d<B: Backend> {
    depthwise_conv: Conv2d<B>,
    pointwise_conv: Conv2d<B>,
}

impl<B: Backend> SeparableConv2d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: [usize; 2],
        device: &Device<B>,
    ) -> Self {
        let depthwise_conv = Conv2dConfig::new([in_channels, in_channels], [3, 3])
            .with_stride(stride)
            .with_padding(PaddingConfig2d::Same)
            .with_groups(in_channels)
            .init(device);

        let pointwise_conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        Self {
            depthwise_conv,
            pointwise_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise_conv.forward(x);
        self.pointwise_conv.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    conv_stem: Conv2d<B>,
    norm_stem: BatchNorm<B, 2>,
    activation: Relu,
    pool_stem: MaxPool2d,
    sep_block1: SeparableConv2d<B>,
    norm1: BatchNorm<B, 2>,
    sep_block2: SeparableConv2d<B>,
    norm2: BatchNorm<B, 2>,
    pool2: MaxPool2d,
    sep_block3: SeparableConv2d<B>,
    norm3: BatchNorm<B, 2>,
    adaptive_pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Cnn<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        let conv_stem = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let norm_stem = BatchNormConfig::new(32).init(device);
        let pool_stem = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let sep_block1 = SeparableConv2d::new(32, 64, [1, 1], device);
        let norm1 = BatchNormConfig::new(64).init(device);

        let sep_block2 = SeparableConv2d::new(64, 128, [1, 1], device);
        let norm2 = BatchNormConfig::new(128).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let sep_block3 = SeparableConv2d::new(128, 256, [1, 1], device);
        let norm3 = BatchNormConfig::new(256).init(device);

        let adaptive_pool = AdaptiveAvgPool2dConfig::new([4, 4]).init();
        let fc1_input_size = 256 * 4 * 4;
        let fc1 = LinearConfig::new(fc1_input_size, 512).init(device);
        let fc2 = LinearConfig::new(512, num_classes).init(device);

        Self {
            conv_stem,
            norm_stem,
            activation: Relu::new(),
            pool_stem,
            sep_block1,
            norm1,
            sep_block2,
            norm2,
            pool2,
            sep_block3,
            norm3,
            adaptive_pool,
            dropout: DropoutConfig::new(0.5).init(),
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.conv_stem.forward(x);
        x = self.norm_stem.forward(x);
        x = self.activation.forward(x);
        x = self.pool_stem.forward(x);

        x = self.sep_block1.forward(x);
        x = self.norm1.forward(x);
        x = self.activation.forward(x);

        x = self.sep_block2.forward(x);
        x = self.norm2.forward(x);
        x = self.activation.forward(x);
        x = self.pool2.forward(x);

        x = self.sep_block3.forward(x);
        x = self.norm3.forward(x);
        x = self.activation.forward(x);

        x = self.adaptive_pool.forward(x);
        let x = x.flatten(1, 3);
        let x = self.dropout.forward(x);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        self.fc2.forward(x)
    }
}
