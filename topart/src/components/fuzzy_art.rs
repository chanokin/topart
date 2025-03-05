use burn::prelude::*;

/// FuzzyIntersection
/// basically an alias to the min_pair function
pub fn fuzzy_intersection<B: Backend, const D: usize>(input_0: Tensor<B, D>, input_1: Tensor<B, 1>) -> Tensor<B, D> {
    input_0.min_pair(input_1)
}

/// TaxicabNorm AKA Manhattan norm/L1 norm
pub fn taxicab_norm<B: Backend>(input: Tensor<B, 2>) -> Tensor<B, 1> {
    input.abs().sum_dim(1).squeeze(1)
}

pub fn choice<B: Backend>(alpha: Float, input: Tensor<B, 2>, weights_i: Tensor<B, 1>) -> Tensor<B, 1> {

    let intersect = fuzzy_intersection(input, weights_i);
    let norm = taxicab_norm(intersect);
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    type TestBackend = burn::backend::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;


    #[test]
    fn num_elements() {
        let device = Default::default();
        let tensor1 = TestTensor::<1>::from_data([1.0, -2.0, 5.0], &device);
        let tensor2 = TestTensor::<1>::from_data([2.0, 3.0, 3.0], &device);
        let tensor = fuzzy_intersection(tensor1, tensor2);
        let expected_tensor = TestTensor::<1>::from_data([1.0, -2.0, 3.0], &device);
        assert!(expected_tensor.all_close(tensor, None, None));
    }
}