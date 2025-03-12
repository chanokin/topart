use burn::prelude::*;
/// ComplementEncoder
/// I'm assuming the input has values between 0 and 1
/// I'll start with the batch-less 1D case
#[derive(Module, Clone, Debug, Default)]
pub struct ComplementEncoder;

impl ComplementEncoder {
    pub fn new() -> Self {
        Self
    }

    // in the forward pass we'll create the new tensor with the complement of the input tensor
    // remember the shape is [batch_size, n_neurons]
    pub fn forward<B: Backend>(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        // note the -input + 1.0 is a weird way of computing the complement of the input tensor
        Tensor::cat(vec![input.clone(), -input.clone() + 1.0], 1)
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
        let sample = TestTensor::<1>::zeros([4], &Default::default());
        let [n_neurons] = sample.dims();
        let sample_n_elements = sample.shape().num_elements();
        assert_eq!(sample_n_elements, 4);

        let encoder = ComplementEncoder::new();
        let output = encoder.forward(sample);
        assert_eq!(output.shape().num_elements(), 2 * sample_n_elements);

        let [n_neurons_i] = output.dims();
        assert_eq!(n_neurons * 2, n_neurons_i);
    }
}