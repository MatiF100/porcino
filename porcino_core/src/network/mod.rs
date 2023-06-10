use ndarray::Array2;

use crate::traits::{Activation, Layer};

use self::{activations::Sigmoid, layers::FFLayer};

mod activations;
mod layers;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<FFLayer>,
}

impl Network {
    pub fn new(neurons: Vec<usize>) -> Self {
        Self {
            layers: neurons
                .windows(2)
                .map(|window| {
                    FFLayer::new(
                        window[0],
                        window[1],
                        crate::enums::InitializationMethods::Zero,
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

    pub fn gradient_descent(&mut self, training_data: Vec<(&Array2<f64>, &Array2<f64>)>, eta: f64) {
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
        for (x, y) in training_data {
            // Getting updated gradients from backpropagation algorithm
            self.process_data(x);
            let (delta_nabla_b, delta_nabla_w) = self.calculate_gradient(x, y);

            // Calculating new gradients with respect to ones created in first steps and also newly calculated ones
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();

            // Something wrong here!!!
            //dbg!(&nabla_w);
            //dbg!(&delta_nabla_w);
            //dbg!(&self);
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
                *w = w.clone() - nw * (eta as f64);
            });

        self.layers
            .iter_mut()
            .map(|layer| &mut layer.biases)
            .zip(nabla_b.iter())
            .for_each(|(w, nw)| {
                *w = w.clone() - nw * (eta as f64);
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
            * &Sigmoid::derivative(&self.layers.last().unwrap().zs, None);

        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = if self.layers.len() >= 2 {
            delta.dot(&self.layers[self.layers.len() - 2].state.t())
        } else {
            delta.dot(&self.layers[0].zs.t())
        };

        let b_len = nabla_b.len();
        let w_len = nabla_w.len();

        // Middle layers
        for (idx, lrs) in self.layers.windows(3).rev().enumerate() {
            let derivative = Sigmoid::derivative(&lrs[1].zs, None);
            delta = &lrs[2].weights.t().dot(&delta) * &derivative;

            nabla_b[b_len - idx - 2] = delta.clone();
            nabla_w[w_len - idx - 2] = delta.dot(&lrs[0].state.t());
        }

        // First layer, if there is more than 1 layer
        if self.layers.len() >= 2 {
            let derivative = Sigmoid::derivative(&self.layers.first().unwrap().zs, None);
            delta = &self.layers[1].weights.t().dot(&delta) * derivative;
            nabla_b[0] = delta.clone();
            nabla_w[0] = delta.dot(&input_set.t());
        }

        (nabla_b, nabla_w)
    }
}
