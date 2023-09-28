use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug, Default)]
pub struct FileView {
    pub headers: Option<Vec<String>>,
    pub fields: Vec<Vec<String>>,
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
