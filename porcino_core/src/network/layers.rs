use crate::{
    enums::InitializationMethods,
    traits::{Activation, Layer},
};
use ndarray::Array2;
use rand::prelude::*;
use rand::distributions::Standard;

use super::activations::Sigmoid;

fn local_sig(x: f64) -> f64{
    1. / ( 1. + (-x).exp())
}

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
                weights: Array2::zeros((neurons, inputs)),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
            },
            InitializationMethods::PseudoSpread => Self {
                weights: Array2::from_shape_fn((neurons, inputs), |(i, j)| {
                    local_sig((i as f64 + 1.0).exp() * (j as f64 + 2.0).ln()) - 0.5
                }),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
            },
            InitializationMethods::Random => Self {
                weights: Array2::from_shape_fn((neurons, inputs), |_| {
                    StdRng::from_entropy().sample(Standard)
                }),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),

            }
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
