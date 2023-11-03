use crate::traits::ErrorFn;

pub struct Sse;
impl ErrorFn for Sse {
    fn cost_function(
        network_output: &ndarray::Array2<f64>,
        reference_set: &ndarray::Array2<f64>,
    ) -> f64 {
        assert_eq!(network_output.raw_dim(), reference_set.raw_dim());
        network_output
            .iter()
            .zip(reference_set)
            .map(|v| (v.0 - v.1))
            .map(|e| e.powf(2.0))
            .sum::<f64>()
    }
}
