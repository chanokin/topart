use burn::prelude::*;
/// ComplementEncoder
/// I'm assuming the input is an image with values between 0 and 1
/// CHANNELS FIRST!
#[derive(Module, Clone, Debug, Default)]
pub struct ComplementEncoder;

impl ComplementEncoder {
    pub fn new() -> Self {
        Self
    }

    // in the forward pass we'll create the new tensor with the complement of the input tensor
    // remember the shape is [batch_size, channels, height, width]
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        Tensor::cat(vec![input.clone(), -input + 1.0], 1)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    type TestBackend = burn::backend::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;


    #[test]
    fn num_elements() {
        // let sample = Tensor::<burn::backend::Wgpu<f32, i32>, 4>::zeros([1, 3, 4, 5], &Default::default());
        let sample = TestTensor::<4>::zeros([1, 3, 4, 5], &Default::default());
        let encoder = ComplementEncoder::new();
        let output = encoder.forward(sample);
        assert_eq!(output.shape().num_elements(), 120);
    }
}