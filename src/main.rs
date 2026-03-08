use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;

use flood_predictor::{DischargeCsvPath, GageHeightCsvPath, StationRecords};

#[derive(Debug, Clone, Parser)]
#[command(
    about,
    subcommand_help_heading = "Commands",
    version
)]
struct Cli {
    /// Path to USGS CSV file containing discharge data of target station.
    #[arg(value_name = "discharge_csv_path")]
    pub discharge_csv_path: PathBuf,

    /// Path to USGS CSV file containing gage height data of target station.
    #[arg(value_name = "gage_height_csv_path")]
    pub gage_height_csv_path: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let discharge_csv_path = DischargeCsvPath::new(cli.discharge_csv_path);
    let gage_height_csv_path = GageHeightCsvPath::new(cli.gage_height_csv_path);
    let records = StationRecords::from_csv_paths(discharge_csv_path, gage_height_csv_path)?;

    println!("{:#?}", records);

    Ok(())
}
