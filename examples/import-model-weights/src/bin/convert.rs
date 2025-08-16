use std::{env, path::Path, process};

use burn::{
    backend::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::PyTorchFileRecorder;
use burn_import::safetensors::SafetensorsFileRecorder;
use import_model_weights::ModelRecord;

// Path constants
const PYTORCH_WEIGHTS_PATH: &str = "weights/mnist.pt";
const SAFETENSORS_WEIGHTS_PATH: &str = "weights/mnist.safetensors";
const MODEL_OUTPUT_NAME: &str = "mnist";

// Basic backend type (not used for computation).
type B = NdArray<f32>;

pub fn main() {
    let args: Vec<String> = env::args().collect();

    // Check argument count
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <pytorch|safetensors> <output_directory>",
            args[0]
        );
        process::exit(1);
    }

    // Get weight format and output directory from arguments
    let weight_format = args[1].as_str();
    let output_directory = Path::new(&args[2]);

    // Use the default device (CPU)
    let device = Default::default();

    // Load the model record based on the specified format
    let model_record: ModelRecord<B> = match weight_format {
        "pytorch" => {
            println!("Loading PyTorch weights from '{PYTORCH_WEIGHTS_PATH}'...");
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(PYTORCH_WEIGHTS_PATH.into(), &device)
                .unwrap_or_else(|_| {
                    panic!("Failed to load PyTorch model weights from '{PYTORCH_WEIGHTS_PATH}'")
                })
        }
        "safetensors" => {
            println!("Loading Safetensors weights from '{SAFETENSORS_WEIGHTS_PATH}'...");
            SafetensorsFileRecorder::<FullPrecisionSettings>::default()
                .load(SAFETENSORS_WEIGHTS_PATH.into(), &device)
                .unwrap_or_else(|_| {
                    panic!(
                        "Failed to load Safetensors model weights from '{SAFETENSORS_WEIGHTS_PATH}'"
                    )
                })
        }
        _ => {
            eprintln!(
                "Error: Unsupported weight format '{weight_format}'. Please use 'pytorch' or 'safetensors'."
            );
            process::exit(1);
        }
    };

    // Create a recorder for saving the model record in Burn's format
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // Define the output path for the Burn model file
    let output_file_path = output_directory.join(MODEL_OUTPUT_NAME);

    println!(
        "Saving model record to '{}.mpk'...",
        output_file_path.display()
    );

    // Save the loaded record to the specified file path
    recorder
        .record(model_record, output_file_path.clone())
        .unwrap_or_else(|_| {
            panic!(
                "Failed to save model record to '{}.mpk'",
                output_file_path.display()
            )
        });

    println!(
        "Model record successfully saved to '{}.mpk'.",
        output_file_path.display()
    );
}
