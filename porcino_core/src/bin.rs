use ndarray::Array2;
use porcino_core::{data, network::Network};
fn main() {
    let mut x = Network::new(vec![13, 8, 3]);
    let t = data::prepare_file("wine.data");
    //let nabla = x.calculate_gradient(&Array2::from_shape_vec((2, 1), vec![1., 1.]).unwrap(),&Array2::from_shape_vec((4, 1), vec![1., 0., 0., 0.]).unwrap());
    /*
        let training_data = vec![
            (
                Array2::from_shape_vec((2, 1), vec![0., 0.]).unwrap(),
                Array2::from_shape_vec((1, 1), vec![0.]).unwrap(),
            ),
            (
                Array2::from_shape_vec((2, 1), vec![0., 1.]).unwrap(),
                Array2::from_shape_vec((1, 1), vec![1.]).unwrap(),
            ),
            (
                Array2::from_shape_vec((2, 1), vec![1., 0.]).unwrap(),
                Array2::from_shape_vec((1, 1), vec![1.]).unwrap(),
            ),
            (
                Array2::from_shape_vec((2, 1), vec![1., 1.]).unwrap(),
                Array2::from_shape_vec((1, 1), vec![0.]).unwrap(),
            ),
        ];
    */

    for _ in 0..10000 {
        //x.gradient_descent(training_data.iter().map(|v| (&v.0, &v.1)).collect(), 0.9);
        x.gradient_descent(t.0.iter().map(|v| (&v.0, &v.1)).collect(), 0.005);
    }

    /*
    x.process_data(&Array2::from_shape_vec((2, 1), vec![0., 0.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![0., 1.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![1., 0.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&Array2::from_shape_vec((2, 1), vec![1., 1.]).unwrap());
    dbg!(&x.layers.last().unwrap().state);
    */

    //dbg!(&t.0[0]);

    // dbg!(&t.0[53]);

    x.process_data(&t.0[40].0);
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&t.0[82].0);
    dbg!(&x.layers.last().unwrap().state);
    x.process_data(&t.0[122].0);
    dbg!(&x.layers.last().unwrap().state);
}
