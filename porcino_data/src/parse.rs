use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

#[derive(Debug, Default)]
pub struct FileView {
    pub headers: Option<Vec<String>>,
    pub fields: Vec<Vec<String>>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct TaggedData {
    pub data: Vec<Vec<f64>>,
    pub meta: Metadata,
}
#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub input: Array2<f64>,
    pub expected_output: Array2<f64>,
}

pub fn get_sampled_data(raw_data: &TaggedData) -> Vec<TrainingSample> {
    raw_data
        .data
        .iter()
        .map(|row| TrainingSample {
            input: Array2::from_shape_vec(
                (raw_data.meta.params.len(), 1),
                row.iter()
                    .enumerate()
                    .filter(|(idx, _)| raw_data.meta.params.contains(idx))
                    .map(|(_, v)| *v)
                    .collect(),
            )
            .unwrap(),
            expected_output: Array2::from_shape_vec(
                (raw_data.meta.classes.len(), 1),
                row.iter()
                    .enumerate()
                    .filter(|(idx, _)| raw_data.meta.classes.contains(idx))
                    .map(|(_, v)| *v)
                    .collect(),
            )
            .unwrap(),
        })
        .collect()
}

pub fn parse_data_file(
    path: &PathBuf,
    settings: &DataSettings,
    header: bool,
    separator: &str,
) -> Result<TaggedData> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = String::new();

    if header {
        reader.read_line(&mut buf)?;
        buf.clear();
    }

    reader.read_to_string(&mut buf)?;

    let intermediate = buf
        .trim()
        .lines()
        .map(|line| line.trim().split(separator).collect())
        .collect::<Vec<_>>();

    let labels = transpose_vec(intermediate.clone(), settings.columns.len())
        .into_iter()
        .enumerate()
        .filter_map(|(idx, column)| match settings.columns[idx] {
            ColumnType::Class(class_type) if matches!(class_type, ClassType::Label) => Some((
                idx,
                column
                    .iter()
                    .fold(
                        (HashMap::<&str, f64>::new(), 0),
                        |(mut working_set, mut counter), record| {
                            if !working_set.contains_key(record) {
                                working_set.insert(record.to_owned(), counter as f64);
                                counter += 1;
                            };
                            (working_set, counter)
                        },
                    )
                    .0,
            )),
            ColumnType::Parameter(parameter_type)
                if matches!(parameter_type, ParameterType::Label) =>
            {
                Some((
                    idx,
                    column
                        .iter()
                        .fold(
                            (HashMap::<&str, f64>::new(), 0),
                            |(mut working_set, mut counter), record| {
                                if !working_set.contains_key(record) {
                                    working_set.insert(record.to_owned(), counter as f64);
                                    counter += 1;
                                };
                                (working_set, counter)
                            },
                        )
                        .0,
                ))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut new_idx = 0;
    let mut meta = Metadata::default();
    let outputs = transpose_vec(intermediate.clone(), settings.columns.len())
        .into_iter()
        .enumerate()
        .filter_map(|(idx, column)| match settings.columns[idx] {
            ColumnType::Class(class_type) => {
                meta.classes.push(new_idx);
                new_idx += 1;
                Some(match class_type {
                    ClassType::Value => column
                        .iter()
                        .map(|v| v.parse::<f64>().unwrap())
                        .collect::<Vec<_>>(),
                    ClassType::Label => column
                        .iter()
                        .map(|v| {
                            let hm = &labels.iter().find(|(c, _)| *c == idx).unwrap().1;
                            *hm.get(v).unwrap()
                        })
                        .collect(),
                })
            }
            ColumnType::Parameter(parameter_type) => {
                meta.params.push(new_idx);
                new_idx += 1;
                Some(match parameter_type {
                    ParameterType::Boolean | ParameterType::NumericUnnormalized => column
                        .iter()
                        .map(|v| v.parse::<f64>().unwrap())
                        .collect::<Vec<_>>(),
                    ParameterType::Numeric => normalize_values(
                        &column
                            .iter()
                            .map(|v| v.parse::<f64>().unwrap())
                            .collect::<Vec<_>>(),
                    ),
                    ParameterType::Label => column
                        .iter()
                        .map(|v| {
                            let hm = &labels.iter().find(|(c, _)| *c == idx).unwrap().1;
                            *hm.get(v).unwrap()
                        })
                        .collect(),
                })
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let data = transpose_vec(outputs, intermediate.len());
    Ok(TaggedData { data, meta })
}
fn transpose_vec<T>(vec: Vec<Vec<T>>, inner_len: usize) -> Vec<Vec<T>> {
    let mut iters: Vec<_> = vec.into_iter().map(|n| n.into_iter()).collect();
    (0..inner_len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect::<Vec<_>>()
}

fn normalize_values(values: &[f64]) -> Vec<f64>
where
{
    let min = values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    values.iter().map(|v| (*v - *min) / (*max - *min)).collect()
}

pub fn get_file_preview(
    path: &PathBuf,
    lines: usize,
    header: bool,
    separator: &str,
) -> Result<FileView> {
    let mut parsed_file = FileView::default();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = String::new();

    if header && reader.read_line(&mut buf).is_ok() {
        parsed_file.headers = Some(buf.trim().split(separator).map(|s| s.to_owned()).collect())
    }
    buf.clear();

    while reader.read_line(&mut buf).is_ok() && !buf.trim().is_empty() {
        parsed_file
            .fields
            .push(buf.trim().split(separator).map(|s| s.to_owned()).collect());
        buf.clear();
        if parsed_file.fields.len() > lines {
            break;
        }
    }
    Ok(parsed_file)
}

#[derive(Default)]
pub struct DataSettings {
    pub columns: Vec<ColumnType>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColumnType {
    Parameter(ParameterType),
    Class(ClassType),
    Ignored,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParameterType {
    Boolean,
    Numeric,
    NumericUnnormalized,
    Label,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ClassType {
    Value,
    Label,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub params: Vec<usize>,
    pub classes: Vec<usize>,
}
