pub mod data;
mod enums;
mod errors;
pub mod network;
mod traits;
/*
//Derivative of sigmoidal function
fn sigmoid_prime<D>(z: Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
{
    let val = sigmoid(z);
    &val * (1.0 - &val)
}

//Implementation of neural network. Includes both, learning algorithms and the usage of network itself
#[derive(Debug, Clone)]
struct Network {
    layers: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
    name: String,
}
*/
