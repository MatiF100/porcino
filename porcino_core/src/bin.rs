use ndarray::Array2;
use porcino_core::enums::InitializationMethods;
use porcino_core::{data, network::Network};
fn main() {
    let mut x = Network::new(vec![13, 8, 3], InitializationMethods::Random);
    let t = data::prepare_file("wine.data", ",");

    for _ in 0..100000 {
        x.gradient_descent(t.0.iter().map(|v| &(&v.0, &v.1)).collect(), 0.0001);
    }

    let result = evaluate(&mut x, &t.0);
    println!("Poprawne dopasowania: {}/{}", result.1, result.0);
}

fn evaluate(net: &mut Network, test_data: &Vec<(Array2<f64>, Array2<f64>)>) -> (usize, usize) {
    let mut local_data = test_data.clone();
    let x = local_data
        .iter_mut()
        .map(|(x, y)| {
            (
                {
                    net.process_data(x);
                    net.layers
                        .last()
                        .unwrap()
                        .state
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(index, _)| index)
                },
                y,
            )
        })
        .filter(|(a, b)| {
            a.unwrap_or(0)
                == b.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
        })
        .count();

    (test_data.len(), x)
}
