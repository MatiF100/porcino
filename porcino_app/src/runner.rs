use porcino_core::errors::Sse;
use porcino_core::network::Network;
use porcino_core::traits::ErrorFn;
use porcino_data::parse::TrainingSample;
use std::sync::mpsc;
use std::thread;
use std::thread::JoinHandle;

pub enum NetworkResponse {
    Epochs(usize, usize),
    EvalResult(Vec<f64>),
}
pub enum NetworkSignal {
    Toggle,
    SetEpochs(usize),
    SetData(Vec<TrainingSample>),
    SetReportInterval(usize),
    EvalData(Option<Vec<TrainingSample>>),
    GetEpochs,
    GetOutput,
}
pub fn run_threaded(
    mut network: Network,
    tx: mpsc::Sender<NetworkResponse>,
    rx: mpsc::Receiver<NetworkSignal>,
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
                    NetworkSignal::GetEpochs => {
                        let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                    }
                    NetworkSignal::GetOutput => {
                        let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                    }
                    NetworkSignal::EvalData(data) => {
                        eval_data = data;
                    }
                    NetworkSignal::SetReportInterval(interval) => report_interval = interval,
                }
            }

            // Network stuff
            if running && epoch_count < epochs_to_run {
                network.gradient_descent(&training_data, eta);
                epoch_count += 1;
                if report_interval != 0 && epoch_count % report_interval == 0 {
                    if let Some(data) = &eval_data {
                        let eval_result = data
                            .iter()
                            .map(|record| {
                                network.process_data(&record.input);
                                let output = &network.layers.last().unwrap().state;
                                Sse::cost_function(output, &record.expected_output)
                            })
                            .collect::<Vec<_>>();
                        let _ = tx.send(NetworkResponse::EvalResult(eval_result));
                    }
                    let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                }
            } else {
                // Send thread to sleep
                let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                resume_message = rx.recv().ok();
            }
        }
    })
}
