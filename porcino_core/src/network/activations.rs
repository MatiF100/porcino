use ndarray::Array2;

use crate::traits::Activation;

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn function(z: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_vec(
            z.raw_dim(),
            z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        )
        .unwrap()
    }

    fn derivative(z: &Array2<f64>, val: Option<&Array2<f64>>) -> Array2<f64> {
        if let Some(val) = val {
            val * (1.0 - val)
        } else {
            let val = Self::function(z);
            &val * (1.0 - &val)
        }
    }
}
