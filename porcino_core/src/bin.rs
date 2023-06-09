use ndarray::Array2;
use porcino_core::network::Network;
fn main() {
    let mut x = Network::new(vec![2, 8, 2, 1]);
    //let nabla = x.calculate_gradient(&Array2::from_shape_vec((2, 1), vec![1., 1.]).unwrap(),&Array2::from_shape_vec((4, 1), vec![1., 0., 0., 0.]).unwrap());
    let training_data = vec![
        (Array2::from_shape_vec((2,1), vec![0., 0.]).unwrap(), Array2::from_shape_vec((1,1), vec![0.]).unwrap()),
        (Array2::from_shape_vec((2,1), vec![0., 1.]).unwrap(), Array2::from_shape_vec((1,1), vec![0.]).unwrap()),
        (Array2::from_shape_vec((2,1), vec![1., 0.]).unwrap(), Array2::from_shape_vec((1,1), vec![0.]).unwrap()),
        (Array2::from_shape_vec((2,1), vec![1., 1.]).unwrap(), Array2::from_shape_vec((1,1), vec![1.]).unwrap())
    ];
    for _ in 0..1000{
        x.gradient_descent(training_data.iter().map(|v| (&v.0, &v.1)).collect(), 2.);
    }
    x.process_data(&Array2::from_shape_vec((2, 1), vec![0.,0.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![0.,1.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![1.,0.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![1.,1.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
}