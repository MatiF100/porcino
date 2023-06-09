use crate::{
    enums::InitializationMethods,
    traits::{Activation, Layer},
};
use ndarray::Array2;

use super::activations::Sigmoid;

#[derive(Debug)]
pub struct FFLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub zs: Array2<f64>,
    pub state: Array2<f64>,
}

impl FFLayer {
    pub fn new(inputs: usize, neurons: usize, weight_init: InitializationMethods) -> Self {
        match weight_init {
            InitializationMethods::Zero => Self {
                weights: Array2::ones((neurons, inputs)),
                biases: Array2::ones((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
            },
            _ => todo!(),
        }
    }
}

impl Layer for FFLayer {
    fn feed_forward(&mut self, input: &ndarray::Array2<f64>) -> &Array2<f64> {
        self.zs = &self.weights.dot(input) + &self.biases;
        self.state = Sigmoid::function(&self.zs);
        &self.state
    }
}
