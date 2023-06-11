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
    let classes: HashMap<String, f64> = records
        .iter()
        .fold(
            (HashMap::<String, f64>::new(), 0),
            |(mut working_set, mut counter), record| {
                if !working_set.contains_key(record.0) {
                    working_set.insert(record.0.to_owned(), counter as f64);
                    counter += 1;
                };
                (working_set, counter)
            },
        )
        .0;

    let maxes = (0..records[0].1.len())
        .map(|i| records.iter().map(|inner| inner.1[i]).max_by(|a,b| a.total_cmp(b)).unwrap())
        //.map(|v| v.1.iter().max_by(|a, b| a.total_cmp(b)).unwrap())
        .collect::<Vec<_>>();
    let mins = (0..records[0].1.len())
        .map(|i| records.iter().map(|inner| inner.1[i]).min_by(|a,b| a.total_cmp(b)).unwrap())
        .collect::<Vec<_>>();

    let parsed = records
        .iter()
        .map(|(class, params)| {
            (
                Array::from_shape_vec(
                    (params.len(), 1),
                    params
                        .iter()
                        .enumerate()
                        .map(|(idx, v)| (v - mins[idx]) / (maxes[idx] - mins[idx]))
                        .collect(),
                )
                .unwrap(),
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

    (parsed, classes)
}
