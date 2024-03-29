use ndarray::Array2;

use crate::traits::Activation;

pub struct Sigmoid;
pub struct Linear;
impl Activation for Sigmoid {
    fn function(&self, z: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_vec(
            z.raw_dim(),
            z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        )
        .unwrap()
    }

    fn derivative(&self, z: &Array2<f64>, val: Option<&Array2<f64>>) -> Array2<f64> {
        if let Some(val) = val {
            val * (1.0 - val)
        } else {
            let val = Self::function(self, z);
            &val * (1.0 - &val)
        }
    }
}
impl Activation for Linear {
    fn function(&self, z: &Array2<f64>) -> Array2<f64> {
        z.clone()
    }

    fn derivative(&self, z: &Array2<f64>, _: Option<&Array2<f64>>) -> Array2<f64> {
        Array2::from_shape_fn(z.raw_dim(), |_| 1.)
    }
}
