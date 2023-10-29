use crate::runner::NetworkResponse::EvalResult;
use ndarray::Array2;
use porcino_core::network::Network;
use porcino_data::parse::TrainingSample;
use std::sync::mpsc;
use std::thread;
use std::thread::JoinHandle;

pub enum NetworkResponse {
    Epochs(usize, usize),
    EvalResult(Vec<Array2<f64>>),
}
pub enum NetworkSignal {
    Start,
    Stop,
    SetEpochs(usize),
    SetData(Vec<TrainingSample>),
    EvalData(Vec<TrainingSample>),
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
        let eta = initial_eta;
        loop {
            // Thread communication
            // This may significantly impact performance
            // Check with profiler later
            if let Ok(signal) = rx.try_recv() {
                match signal {
                    NetworkSignal::Start => running = true,
                    NetworkSignal::SetEpochs(epochs) => epochs_to_run += epochs,
                    NetworkSignal::Stop => running = false,
                    NetworkSignal::SetData(data) => training_data = data.clone(),
                    NetworkSignal::GetEpochs => {
                        let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                    }
                    NetworkSignal::GetOutput => {
                        let _ = tx.send(NetworkResponse::Epochs(epoch_count, epochs_to_run));
                    }
                    NetworkSignal::EvalData(data) => {
                        let eval_result = data
                            .iter()
                            .map(|record| {
                                network.process_data(&record.input);
                                let output = &network.layers.last().unwrap().state;
                                output - &record.expected_output
                            })
                            .collect::<Vec<_>>();
                        let _ = tx.send(EvalResult(eval_result));
                    }
                }
            }
            // Network stuff
            if running && epoch_count < epochs_to_run {
                network.gradient_descent(&training_data, eta);
                epoch_count += 1;
            }
        }
    })
}
