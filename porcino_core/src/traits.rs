use ndarray::Array2;
pub trait Activation {
    fn function(z: &Array2<f64>) -> Array2<f64>;
    fn derivative(z: &Array2<f64>, val: Option<&Array2<f64>>) -> Array2<f64>;
}

pub trait Layer {
    fn feed_forward(&mut self, input: &Array2<f64>) -> &Array2<f64>;
}

pub trait ErrorFn {
    fn cost_function(&self, data: &Array2<f64>, reference: &Array2<f64>) -> f64;
}
