use crate::runner::{run_threaded, NetworkResponse, NetworkSignal};
use egui::{Color32, Slider};
use egui_file::FileDialog;
use porcino_core::enums::InitializationMethods;
use porcino_core::network::Activations::{Linear, Sigmoid};
use porcino_core::network::{LayerSettings, Network};
use porcino_data::parse::{get_sampled_data, parse_data_file, ClassType, FileView, TaggedData};
use porcino_data::parse::{ColumnType, DataSettings, ParameterType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::channel;

pub struct TemplateApp {
    current_panel: Panels,
    opened_file: Option<PathBuf>,
    opened_file_dialog: Option<FileDialog>,
    has_headers: bool,
    separator: String,
    preview_lines: usize,
    file_preview: Option<PreviewData>,
    data_settings: DataSettings,
    dataset: Option<TaggedData>,
    network: Option<(mpsc::Receiver<NetworkResponse>, mpsc::Sender<NetworkSignal>)>,
    net_conf: NetPreConfig,
}

#[derive(Debug)]
enum PreviewData {
    Ok(FileView),
    Err(String),
}
#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum Panels {
    Landing,
    Data,
    Network,
    Visualize,
}
#[derive(Default)]
struct NetPreConfig {
    layers: Vec<usize>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            current_panel: Panels::Landing,
            opened_file_dialog: None,
            opened_file: None,
            has_headers: false,
            separator: String::from(";"),
            preview_lines: 5,
            file_preview: None,
            data_settings: DataSettings::default(),
            dataset: None,
            network: None,
            net_conf: NetPreConfig::default(),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.

        Default::default()
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Self {
            current_panel,
            opened_file,
            opened_file_dialog,
            has_headers,
            separator,
            preview_lines,
            file_preview,
            data_settings,
            dataset,
            network,
            net_conf,
        } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        _frame.close();
                    }
                });
                ui.menu_button("Data", |ui| {
                    if ui.button("Load file").clicked() {
                        // Open file dialog
                        let mut dialog = FileDialog::open_file(opened_file.clone());
                        dialog.open();
                        *opened_file_dialog = Some(dialog);
                    }
                    if ui.button("Unload file").clicked() {
                        *opened_file = None;
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Select panel");
            if ui.button("Home").clicked() {
                *current_panel = Panels::Landing;
            }
            if ui.button("Data parser").clicked() {
                *current_panel = Panels::Data;
            }
            if ui.button("Network configurator").clicked() {
                *current_panel = Panels::Network;
            }
            if ui.button("Network visualizer").clicked() {
                *current_panel = Panels::Visualize;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's

            if let Some(dialog) = opened_file_dialog {
                if dialog.show(ctx).selected() {
                    if let Some(file) = dialog.path() {
                        *opened_file = Some(PathBuf::from(file));
                    }
                }
            }
            match current_panel {
                Panels::Landing => {
                    ui.heading("eframe template");
                    ui.hyperlink("https://github.com/emilk/eframe_template");
                    ui.add(egui::github_link_file!(
                        "https://github.com/emilk/eframe_template/blob/master/",
                        "Source code."
                    ));
                }
                Panels::Data => {
                    if ui.button("Load file").clicked() {
                        // Open file dialog
                        let mut dialog = FileDialog::open_file(opened_file.clone());
                        dialog.open();
                        *opened_file_dialog = Some(dialog);
                    }
                    if let Some(file) = opened_file {
                        ui.label(format!(
                            "Currently selected file: {}",
                            file.to_str().unwrap()
                        ));
                        ui.collapsing("Preview and configure", |ui| {
                            ui.checkbox(has_headers, "Headers").on_hover_text(
                                "Whether file contains column names in the first row",
                            );
                            ui.horizontal(|ui| {
                                ui.label("Separator:");
                                ui.add(
                                    egui::TextEdit::singleline(separator)
                                        .hint_text("Enter the separator string for values"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("Preview lines:");
                                ui.add(egui::DragValue::new(preview_lines));
                            });
                            if ui.button("Show preview").clicked() {
                                // Initial data parsing
                                let parsed_file = porcino_data::parse::get_file_preview(
                                    file,
                                    *preview_lines,
                                    *has_headers,
                                    separator.as_str(),
                                );

                                match parsed_file {
                                    Ok(f) => {
                                        data_settings.columns = vec![
                                            ColumnType::Ignored;
                                            f.fields.first().unwrap().len()
                                        ];
                                        *file_preview = Some(PreviewData::Ok(f));
                                    }
                                    Err(e) => *file_preview = Some(PreviewData::Err(e.to_string())),
                                }
                            }
                            if let Some(preview) = file_preview {
                                match preview {
                                    PreviewData::Ok(data) => {
                                        let labels = if let Some(headers) = &data.headers {
                                            headers.to_owned()
                                        } else {
                                            data.fields
                                                .first()
                                                .unwrap()
                                                .iter()
                                                .enumerate()
                                                .map(|(i, _)| format!("Field {}", i))
                                                .collect()
                                        };

                                        egui::ScrollArea::both().show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                labels.iter().enumerate().for_each(|(i, label)| {
                                                    ui.vertical(|ui| {
                                                        ui.set_min_width((label.len() * 7) as f32);
                                                        ui.menu_button(label, |ui| {
                                                            ui.selectable_value(
                                                                &mut data_settings.columns[i],
                                                                ColumnType::Ignored,
                                                                "Ignored",
                                                            );
                                                            ui.menu_button("Parameter", |ui| {
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Parameter(
                                                                        ParameterType::Boolean,
                                                                    ),
                                                                    "Boolean",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Parameter(
                                                                        ParameterType::Label,
                                                                    ),
                                                                    "Text label",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Parameter(
                                                                        ParameterType::Numeric,
                                                                    ),
                                                                    "Number",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Parameter(
                                                                        ParameterType::NumericUnnormalized,
                                                                    ),
                                                                    "Number (not normalized)",
                                                                );
                                                            });
                                                            ui.menu_button("Class", |ui| {
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Class(
                                                                        ClassType::Label
                                                                    ),
                                                                    "Text label",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut data_settings.columns[i],
                                                                    ColumnType::Class(
                                                                        ClassType::Value
                                                                    ),
                                                                    "Number",
                                                                );
                                                            });
                                                        });
                                                        data.fields.iter().for_each(|row| {
                                                            ui.label(row[i].clone());
                                                        })
                                                    });
                                                });
                                            });
                                        });

                                        if ui.button("PARSE!").clicked(){
                                            let parsed_data = parse_data_file(file, data_settings, *has_headers, separator);
                                            if let Ok(v) = parsed_data{
                                                *dataset = Some(v);
                                            }
                                        }
                                    }
                                    PreviewData::Err(e) => {
                                        ui.colored_label(egui::Color32::RED, e);
                                    }
                                };
                            }
                        });
                    }
                }
                Panels::Network => {
                    if let Some(dataset) = dataset{
                        if ui.button("Add layer").clicked(){
                            net_conf.layers.push(0);
                        }

                        ui.label(format!("Input neurons: {}", dataset.meta.params.len()));
                        for layer in &mut net_conf.layers{
                            ui.add(Slider::new(layer, 0usize..=100usize).text("Neurons"));
                        }
                        ui.label(format!("Output neurons: {}", dataset.meta.classes.len()));

                        if ui.button("Generate Network structure").clicked(){
                            let mut total_layers = net_conf.layers.iter().map(|v| LayerSettings{neurons: *v, activation: Sigmoid}).collect::<Vec<_>>();
                            total_layers.insert(0, LayerSettings{neurons: dataset.meta.params.len(), activation: Linear});
                            total_layers.push(LayerSettings{neurons: dataset.meta.classes.len(), activation: Linear});
                            let local_network = Network::new(total_layers, InitializationMethods::Random);
                            let signals = channel::<NetworkSignal>();
                            let responses = channel::<NetworkResponse>();

                            let _ = run_threaded(local_network, responses.0, signals.1, 0.0001);
                            *network = Some((responses.1, signals.0));
                        }
                    }else{
                        ui.colored_label(Color32::DARK_RED, "No active dataset! Cannot infer network options");
                    }
                }
                Panels::Visualize => {
                    if let Some((rx, tx)) = network{
                        if let Some(data) = dataset{
                            if ui.button("Conf dataset").clicked(){
                                let _ = tx.send(NetworkSignal::SetData(get_sampled_data(data)));
                            }
                            if ui.button("Conf epochs").clicked(){
                                let _ = tx.send(NetworkSignal::SetEpochs(3000));
                            }
                            if ui.button("Start").clicked(){
                                let _ = tx.send(NetworkSignal::Start);
                            }
                            if ui.button("Eval").clicked(){
                                let _ = tx.send(NetworkSignal::EvalData(get_sampled_data(data)));
                                if let Ok(NetworkResponse::EvalResult(res)) = rx.recv(){
                                    dbg!(res);
                                };
                            }
                        }
                    }

                }
                _ => {}
            }

            egui::warn_if_debug_build(ui);
        });

        egui::SidePanel::right("right_panel").show(ctx, |ui| {
            ui.heading("Utility panel");
        });
    }
}
