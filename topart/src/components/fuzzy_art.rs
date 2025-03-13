use burn::{backend::wgpu::FloatElement, prelude::*};

/// FuzzyIntersection
/// basically an alias to the min_pair function
pub fn fuzzy_intersection<B: Backend, const D: usize>(input_0: Tensor<B, D>, input_1: Tensor<B, D>) -> Tensor<B, D> {
    input_0.min_pair(input_1)
}

/// TaxicabNorm AKA Manhattan norm/L1 norm
pub fn taxicab_norm<B: Backend>(input: Tensor<B, 1>) -> Tensor<B, 1> {
    input.abs().sum_dim(0)
}

/// Choice
/// This is the choice function for the Fuzzy ART algorithm
/// It takes the input tensor, the weights tensor, and the vigilance parameter alpha
/// It returns the choice tensor which is the "fitness" of an output neuron[i] to the input (x weights[i])
pub fn choice<B: Backend>(alpha: f32, input: Tensor<B, 1>, weights_i: Tensor<B, 1>) -> Tensor<B, 1> {

    let intersect = fuzzy_intersection(input, weights_i.clone());
    let inter_norm = taxicab_norm(intersect);
    let weights_norm = taxicab_norm(weights_i);
    inter_norm / (weights_norm + alpha)
}

pub fn resonates<B: Backend>(rho: f32, input: Tensor<B, 1>, weights_i: Tensor<B, 1>) -> bool {
    let fitness = choice(0.0, input, weights_i);
    fitness.greater_equal_elem(rho).into_scalar()
}


pub fn weight_update<B: Backend>(beta: f32, input: Tensor<B, 1>, weights_i: Tensor<B, 1>) -> Tensor<B, 1> {
    weights_i.clone() * (1.0 - beta) + fuzzy_intersection(input, weights_i) * beta
}

// /// Category size
// /// This function returns the number of neurons in the category layer
// pub fn category_size<B: Backend>(output_index: usize, weights: Tensor<B, 2>) -> Float {
//     let [out_dim, in_dim] = weights.dims();

//     1.0
// }

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    type TestBackend = burn::backend::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;


    #[test]
    fn intersection() {
        let device = Default::default();
        let tensor1 = TestTensor::<1>::from_data([1.0, -2.0, 5.0], &device);
        let tensor2 = TestTensor::<1>::from_data([2.0, 3.0, 3.0], &device);
        let tensor = fuzzy_intersection(tensor1, tensor2);
        let expected_tensor = TestTensor::<1>::from_data([1.0, -2.0, 3.0], &device);
        assert!(expected_tensor.all_close(tensor, None, None));
    }

    #[test]
    fn norm() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data([1.0, -2.0, 3.0], &device);
        let norm = taxicab_norm(tensor);
        let expected_norm = TestTensor::<1>::from_data([6.0], &device);
        assert!(expected_norm.all_close(norm, None, None));
    }
}