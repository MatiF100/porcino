use crate::runner::{run_threaded, NetworkHandles, NetworkResponse, NetworkSignal};
use egui::{Color32, DragValue, ProgressBar, RichText, Slider};
use egui_file::FileDialog;
use porcino_core::enums::InitializationMethods;
use porcino_core::network::Activations::{Linear, Sigmoid};
use porcino_core::network::{LayerSettings, Network};
use porcino_data::parse::{get_sampled_data, parse_data_file, ClassType, FileView, TaggedData};
use porcino_data::parse::{ColumnType, DataSettings, ParameterType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::{Arc, RwLock};

#[derive(Debug, Default)]
struct NetworkInfo {}
pub struct PorcinoApp {
    current_panel: Panels,
    opened_file: Option<PathBuf>,
    opened_file_dialog: Option<FileDialog>,
    save_data_dialog: Option<FileDialog>,
    load_data_dialog: Option<FileDialog>,
    has_headers: bool,
    separator: String,
    preview_lines: usize,
    file_preview: Option<PreviewData>,
    data_settings: DataSettings,
    dataset: Option<TaggedData>,
    active_networks: Vec<NetworkHandles>,
    network_info: Arc<RwLock<NetworkInfo>>,
    selected_network: usize,
    net_conf: NetPreConfig,
    read_progress: bool,
    progress: f32,
    total_sse: f64,
    report_interval: usize,
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

impl Default for PorcinoApp {
    fn default() -> Self {
        Self {
            current_panel: Panels::Landing,
            opened_file_dialog: None,
            save_data_dialog: None,
            load_data_dialog: None,
            opened_file: None,
            has_headers: false,
            separator: String::from(";"),
            preview_lines: 5,
            file_preview: None,
            data_settings: DataSettings::default(),
            dataset: None,
            active_networks: vec![],
            selected_network: 0,
            network_info: Arc::new(RwLock::new(NetworkInfo::default())),
            net_conf: NetPreConfig::default(),
            read_progress: false,
            progress: 0.0,
            total_sse: 0.0,
            report_interval: 0,
        }
    }
}

impl PorcinoApp {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.

        Default::default()
    }
}

impl eframe::App for PorcinoApp {
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
            active_networks,
            selected_network,
            network_info,
            net_conf,
            read_progress,
            progress,
            total_sse,
            report_interval,
            save_data_dialog,
            load_data_dialog,
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
            if let Some(dialog) = save_data_dialog{
                if dialog.show(ctx).selected() {
                    if let Some(data) = dataset {
                        if let Some(file) = dialog.path() {
                            porcino_data::persistence::save(data, &file.to_path_buf()).unwrap();
                        }
                    }
                }
            }
            if let Some(dialog) = load_data_dialog{
                if dialog.show(ctx).selected() {
                    if let Some(file) = dialog.path() {
                        *dataset = Some(porcino_data::persistence::read( &file.to_path_buf()).unwrap());
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

                            let handle = run_threaded(local_network, responses.0, signals.1, 0.0001);
                            active_networks.push(
                                NetworkHandles{
                                    thread_handler: handle,
                                    tx_handle: signals.0,
                                    rx_handle: responses.1
                                });
                        }
                    }else{
                    ui.colored_label(Color32::DARK_RED, "No active dataset! Cannot infer network options");
                    }
                }
                Panels::Visualize => {
                    if let Some(handles) = active_networks.get(*selected_network){
                        // Send signal
                        if let Some(data) = dataset{
                            if ui.button("Conf dataset").clicked(){
                                let _ = handles.tx_handle.send(NetworkSignal::SetData(get_sampled_data(data)));
                            }
                            if ui.button("Conf epochs").clicked(){
                                let _ = handles.tx_handle.send(NetworkSignal::SetEpochs(3000));
                            }
                            ui.add(Slider::new(report_interval, 0usize..=100usize).text("Report interval (epochs)"));
                            if ui.button("Set report interval").clicked(){
                                let _ = handles.tx_handle.send(NetworkSignal::SetReportInterval(*report_interval));
                            }
                            if ui.button("Toggle learning process").clicked(){
                                let _ = handles.tx_handle.send(NetworkSignal::Toggle);
                            }
                            if ui.button("Toggle evaluation").clicked(){
                                let _ = handles.tx_handle.send(NetworkSignal::EvalData(Some(get_sampled_data(data))));
                            }
                            if ui.button(RichText::new("Toggle status read").color(if *read_progress {Color32::LIGHT_GREEN} else {Color32::LIGHT_GRAY})).clicked(){
                                *read_progress = !*read_progress;
                            }
                        }
                        // Try recieve signal
                        if let Ok(sig) = handles.rx_handle.try_recv(){
                            match sig{
                                NetworkResponse::Epochs(passed, expected) => *progress = passed as f32 / expected as f32,
                                NetworkResponse::EvalResult(eval_results) => *total_sse = eval_results.iter().map(|v| v*v).sum(),
                            }
                        }
                    }

                    ui.add(ProgressBar::new(*progress).show_percentage().fill(if *read_progress{Color32::BLUE } else{Color32::LIGHT_RED}).desired_width(100.0).animate(*read_progress));
                    ui.add_enabled(false, DragValue::new(total_sse));
                }
            }

            egui::warn_if_debug_build(ui);
        });

        egui::SidePanel::right("right_panel").show(ctx, |ui| {
            ui.heading("Dataset");
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    // Open file dialog
                    let mut dialog = FileDialog::save_file(Some(PathBuf::new()));
                    dialog.open();
                    *save_data_dialog = Some(dialog);
                }
                if ui.button("Load").clicked() {
                    // Open file dialog
                    let mut dialog = FileDialog::open_file(Some(PathBuf::new()));
                    dialog.open();
                    *load_data_dialog = Some(dialog);
                }
            });
            if dataset.is_some() {
                ui.label("Dataset active!");
            }
            ui.separator();
            // Active networks module
            ui.heading("Running networks");
            let mut del_net = None;
            active_networks
                .iter()
                .enumerate()
                .for_each(|(idx, network)| {
                    let label = if let Some(label) = network.thread_handler.thread().name() {
                        label.to_owned()
                    } else {
                        format!("Network {}:", idx + 1)
                    };
                    ui.horizontal(|ui| {
                        ui.label(label);
                        if ui.button("Select").clicked() {
                            *selected_network = idx;
                        }
                        if ui.button("Delete").clicked() {
                            del_net = Some(idx);
                        }
                    });
                });

            if let Some(idx) = del_net {
                if let Some(net) = active_networks.get(idx) {
                    let _ = net.tx_handle.send(NetworkSignal::Kill);
                }
                active_networks.remove(idx);
            }
        });
    }
}
