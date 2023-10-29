use crate::{
    enums::InitializationMethods,
    traits::{Activation, Layer},
};
use ndarray::Array2;
use rand::distributions::Standard;
use rand::prelude::*;
use std::fmt::{Debug, Formatter};

fn local_sig(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub struct FFLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub zs: Array2<f64>,
    pub state: Array2<f64>,
    pub activation: &'static (dyn Activation + Send + Sync),
}
impl Debug for FFLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.weights))
    }
}

impl FFLayer {
    pub fn new(
        inputs: usize,
        neurons: usize,
        weight_init: InitializationMethods,
        activation: &'static (dyn Activation + Send + Sync),
    ) -> Self {
        match weight_init {
            InitializationMethods::Zero => Self {
                weights: Array2::zeros((neurons, inputs)),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
                activation,
            },
            InitializationMethods::One => Self {
                weights: Array2::ones((neurons, inputs)),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
                activation,
            },
            InitializationMethods::PseudoSpread => Self {
                weights: Array2::from_shape_fn((neurons, inputs), |(i, j)| {
                    local_sig((i as f64 + 1.0).exp() * (j as f64 + 2.0).ln()) - 0.5
                }),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
                activation,
            },
            InitializationMethods::Random => Self {
                weights: Array2::from_shape_fn((neurons, inputs), |_| {
                    StdRng::from_entropy().sample(Standard)
                }),
                biases: Array2::zeros((neurons, 1)),
                zs: Array2::zeros((neurons, 1)),
                state: Array2::zeros((neurons, 1)),
                activation,
            },
        }
    }
}

impl Layer for FFLayer {
    fn feed_forward(&mut self, input: &ndarray::Array2<f64>) -> &Array2<f64> {
        self.zs = &self.weights.dot(input) + &self.biases;
        self.state = self.activation.function(&self.zs);
        &self.state
    }
}
