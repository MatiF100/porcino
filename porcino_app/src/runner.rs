use crate::app::NetworkInfo;
use porcino_core::errors::Sse;
use porcino_core::network::Network;
use porcino_core::traits::ErrorFn;
use porcino_data::parse::TrainingSample;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::thread::JoinHandle;

pub struct NetworkHandles {
    pub thread_handler: JoinHandle<()>,
    pub rx_handle: mpsc::Receiver<NetworkResponse>,
    pub tx_handle: mpsc::Sender<NetworkSignal>,
}
pub enum NetworkResponse {}
pub enum NetworkSignal {
    Toggle,
    Kill,
    SetEpochs(usize),
    SetData(Vec<TrainingSample>),
    SetReportInterval(usize),
    EvalData(Option<Vec<TrainingSample>>),
}
pub fn run_threaded(
    mut network: Network,
    tx: mpsc::Sender<NetworkResponse>,
    rx: mpsc::Receiver<NetworkSignal>,
    status: Arc<RwLock<NetworkInfo>>,
    initial_eta: f64,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // Basically state variables
        // May be a good idea to move these into a struct
        let mut running = false;
        let mut epoch_count = 0;
        let mut epochs_to_run = 0;
        let mut training_data = Vec::new();
        let mut eval_data = None;
        let mut report_interval = 0;
        let mut resume_message: Option<NetworkSignal> = None;
        let mut eval_result: f64 = 0.0;
        let eta = initial_eta;
        loop {
            // Thread communication
            // This may significantly impact performance
            // Check with profiler later
            let signal = if let Some(sig) = resume_message {
                resume_message = None;
                Ok(sig)
            } else {
                rx.try_recv()
            };
            if let Ok(signal) = signal {
                match signal {
                    NetworkSignal::Toggle => running = !running,
                    NetworkSignal::SetEpochs(epochs) => epochs_to_run += epochs,
                    NetworkSignal::SetData(data) => training_data = data.clone(),
                    NetworkSignal::EvalData(data) => {
                        eval_data = data;
                    }
                    NetworkSignal::SetReportInterval(interval) => report_interval = interval,
                    NetworkSignal::Kill => break,
                }
            } else {
                if report_interval != 0 && epoch_count % report_interval == 0 {
                    if let Some(data) = &eval_data {
                        eval_result = data
                            .iter()
                            .map(|record| {
                                network.process_data(&record.input);
                                let output = &network.layers.last().unwrap().state;
                                Sse::cost_function(output, &record.expected_output)
                            })
                            .map(|v| v * v)
                            .sum();
                    }

                    report_status(
                        status.clone(),
                        epoch_count,
                        epochs_to_run,
                        eval_result,
                        running,
                    );
                }

                // Network stuff
                if running && epoch_count < epochs_to_run {
                    network.gradient_descent(&training_data, eta);
                    epoch_count += 1;
                } else {
                    // Send thread to sleep
                    report_status(
                        status.clone(),
                        epoch_count,
                        epochs_to_run,
                        eval_result,
                        false,
                    );
                    resume_message = rx.recv().ok();
                }
            }
        }
    })
}

fn report_status(
    lock: Arc<RwLock<NetworkInfo>>,
    epochs: usize,
    epochs_to_run: usize,
    last_eval_result: f64,
    running: bool,
) {
    if let Ok(mut guard) = lock.write() {
        guard.epochs = epochs;
        guard.epochs_to_run = epochs_to_run;
        guard.last_eval_result = last_eval_result;
        guard.running = running;
    }
}
