use egui_file::FileDialog;
use porcino_data::parse::FileView;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    // Example stuff:
    current_panel: Panels,
    opened_file: Option<PathBuf>,
    opened_file_dialog: Option<FileDialog>,
    has_headers: bool,
    separator: String,
    preview_lines: usize,
    file_preview: Option<PreviewData>,
    data_settings: DataSettings,
}

#[derive(Debug)]
enum PreviewData {
    Ok(FileView),
    Err(String),
}
#[derive(Default)]
struct DataSettings {
    columns: Vec<ColumnType>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ColumnType {
    Parameter(ParameterType),
    Class(ClassType),
    Ignored,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ParameterType {
    Boolean,
    Numeric,
    NumericUnnormalized,
    Label,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ClassType {
    Value,
    Label,
}
#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum Panels {
    Landing,
    Data,
    Network,
    Visualize,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            // Example stuff:
            current_panel: Panels::Landing,
            opened_file_dialog: None,
            opened_file: None,
            has_headers: false,
            separator: String::from(";"),
            preview_lines: 5,
            file_preview: None,
            data_settings: DataSettings::default(),
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
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            if ui.button("Load file").clicked() {
                // Open file dialog
                let mut dialog = FileDialog::open_file(opened_file.clone());
                dialog.open();
                *opened_file_dialog = Some(dialog);
            }

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
                                    }
                                    PreviewData::Err(e) => {
                                        ui.colored_label(egui::Color32::RED, e);
                                    }
                                };
                            }
                        });
                    }
                }
                _ => {}
            }

            egui::warn_if_debug_build(ui);
        });
    }
}
