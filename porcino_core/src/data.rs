use std::collections::HashMap;

use ndarray::{Array, Array2};

pub fn prepare_file(filename: &str) -> (Vec<(Array2<f64>, Array2<f64>)>, HashMap<String, f64>) {
    let contents = std::fs::read_to_string(filename).unwrap();
    let records = contents
        .trim()
        .lines()
        .map(|line| line.trim().rsplit_once(",").unwrap())
        .map(|(params, class)| {
            (
                class,
                params
                    .split(",")
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect(),
            )
        })
        .collect::<Vec<(&str, Vec<f64>)>>();
    let classes: HashMap<&str, f64> = records
        .iter()
        .fold(
            (HashMap::<&str, f64>::new(), 0),
            |(mut working_set, mut counter), record| {
                if !working_set.contains_key(record.0){
                    working_set.insert(record.0, counter as f64);
                    counter += 1;
                };
                (working_set, counter)
            },
        )
        .0;

    let parsed = records
        .iter()
        .map(|(class, params)| {
            (
                Array::from_shape_vec((params.len(), 1), params.clone()).unwrap(),
                Array::from_shape_vec(
                    (classes.len(), 1),
                    (0..classes.len())
                        .map(|x| {
                            if *classes.get(class.to_owned()).unwrap() == x as f64 {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    (parsed, HashMap::new())
}
