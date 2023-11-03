use crate::network::activations::Linear;
use ndarray::Array2;
use porcino_data::parse::TrainingSample;

use crate::traits::{Activation, Layer};

use self::{activations::Sigmoid, layers::FFLayer};

mod activations;
mod layers;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<FFLayer>,
}

pub struct LayerSettings {
    pub neurons: usize,
    pub activation: Activations,
}

pub enum Activations {
    Sigmoid,
    Linear,
}

impl Network {
    pub fn new(neurons: Vec<LayerSettings>, init: crate::enums::InitializationMethods) -> Self {
        Self {
            layers: neurons
                .windows(2)
                .map(|window| {
                    FFLayer::new(
                        window[0].neurons,
                        window[1].neurons,
                        init,
                        match window[1].activation {
                            Activations::Sigmoid => &Sigmoid,
                            Activations::Linear => &Linear,
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn process_data(&mut self, input: &Array2<f64>) {
        let mut input = self.layers[0].feed_forward(input).clone();
        for i in 1..self.layers.len() {
            input = self.layers[i].feed_forward(&input).clone();
        }
    }

    pub fn gradient_descent(&mut self, training_data: &Vec<TrainingSample>, eta: f64) {
        // Allocation of gradient vectors
        let mut nabla_b = self
            .layers
            .iter()
            .map(|layer| &layer.biases)
            .map(|b| Array2::zeros(b.raw_dim()))
            .collect::<Vec<Array2<f64>>>();
        let mut nabla_w = self
            .layers
            .iter()
            .map(|layer| &layer.weights)
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect::<Vec<Array2<f64>>>();

        // Loop performing learning iteration over all mini_batches
        for sample in training_data {
            // Getting updated gradients from backpropagation algorithm
            self.process_data(&sample.input);
            let (delta_nabla_b, delta_nabla_w) =
                self.calculate_gradient(&sample.input, &sample.expected_output);

            // Calculating new gradients with respect to ones created in first steps and also newly calculated ones
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();

            nabla_w = nabla_w
                .iter()
                .zip(delta_nabla_w.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }

        // Calculating new values for weights and biases based on recieved gradients with respect to batch size and learning rate
        self.layers
            .iter_mut()
            .map(|layer| &mut layer.weights)
            .zip(nabla_w.iter())
            .for_each(|(w, nw)| {
                *w = w.clone() - nw * eta;
            });

        self.layers
            .iter_mut()
            .map(|layer| &mut layer.biases)
            .zip(nabla_b.iter())
            .for_each(|(w, nw)| {
                *w = w.clone() - nw * eta;
            });
    }
    pub fn calculate_gradient(
        &self,
        input_set: &Array2<f64>,
        reference_set: &Array2<f64>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b = self
            .layers
            .iter()
            .map(|layer| &layer.biases)
            .map(|b| Array2::zeros(b.raw_dim()))
            .collect::<Vec<Array2<f64>>>();
        let mut nabla_w = self
            .layers
            .iter()
            .map(|layer| &layer.weights)
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect::<Vec<Array2<f64>>>();

        // Last layer
        let mut delta = (&self.layers.last().unwrap().state - reference_set)
            * &self
                .layers
                .last()
                .unwrap()
                .activation
                .derivative(&self.layers.last().unwrap().zs, None);

        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = if self.layers.len() >= 2 {
            delta.dot(&self.layers[self.layers.len() - 2].state.t())
        } else {
            delta.dot(&input_set.t())
        };

        let b_len = nabla_b.len();
        let w_len = nabla_w.len();

        // Middle layers
        for (idx, lrs) in self.layers.windows(3).rev().enumerate() {
            let derivative = lrs[1].activation.derivative(&lrs[1].zs, None);
            delta = &lrs[2].weights.t().dot(&delta) * &derivative;

            nabla_b[b_len - idx - 2] = delta.clone();
            nabla_w[w_len - idx - 2] = delta.dot(&lrs[0].state.t());
        }

        // First layer, if there is more than 1 layer
        if self.layers.len() >= 2 {
            let derivative = self
                .layers
                .first()
                .unwrap()
                .activation
                .derivative(&self.layers[0].zs, None);
            delta = &self.layers[1].weights.t().dot(&delta) * derivative;
            nabla_b[0] = delta.clone();
            nabla_w[0] = delta.dot(&input_set.t());
        }

        (nabla_b, nabla_w)
    }
}
