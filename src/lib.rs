use serde::{Deserialize, Deserializer};
use std::{
    fmt,
    path::{Path, PathBuf},
};

/// USGS parameter code.
///
/// USGS documents the kind of data they measured via _parameter codes_. All
/// parameter codes follow a five digit format meant to describe specific
/// constituents and their unit of measure, e.g., chemicals, water discharge,
/// water gage height, etc.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum ParameterCode {
    /// Water discharge constituent.
    ///
    /// The amount of water a stream (or lake) discharged over a period of time.
    /// Measured in cubic feet per second.
    #[default]
    Discharge = 60,

    /// Water gage height constituent.
    ///
    /// The height of the water surface relative to a reference datum. Measured
    /// in feet.
    GageHeight = 65,
}

impl fmt::Display for ParameterCode {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discharge => write!(fmt, "00060"),
            Self::GageHeight => write!(fmt, "00065"),
        }
    }
}

fn to_parameter_code_variant<'de, D>(deserializer: D) -> Result<ParameterCode, D::Error>
where
    D: Deserializer<'de>,
{
    let code: String = String::deserialize(deserializer)?;

    match code.as_ref() {
        "00060" => Ok(ParameterCode::Discharge),
        "00065" => Ok(ParameterCode::GageHeight),
        _ => Err(serde::de::Error::custom(Error::UnknownParameterCode {
            code,
        })),
    }
}

/// Raw CSV Record.
///
/// USGS follows a consistent format for their CSV files. We mainly care about
/// the _time_, _value_, and _parameter code_, columns.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct RawRecord {
    /// Timestamp when data point was collected.
    ///
    /// USGS follows the ISO/IEC 8601 time format.
    pub time: String,

    /// The recorded value collected.
    pub value: f32,

    /// Parameter code of record.
    #[serde(deserialize_with = "to_parameter_code_variant")]
    pub parameter_code: ParameterCode,
}

impl RawRecord {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Path to discharge CSV file.
#[derive(Debug, Default, Clone)]
pub struct DischargeCsvPath(pub(crate) PathBuf);

impl DischargeCsvPath {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self(path.as_ref().to_path_buf())
    }

    pub fn as_path(&self) -> &Path {
        self.0.as_path()
    }

    pub fn to_path_buf(&self) -> PathBuf {
        self.0.clone()
    }
}

/// Path to gage height CSV file.
#[derive(Debug, Default, Clone)]
pub struct GageHeightCsvPath(pub(crate) PathBuf);

impl GageHeightCsvPath {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self(path.as_ref().to_path_buf())
    }

    pub fn as_path(&self) -> &Path {
        self.0.as_path()
    }

    pub fn to_path_buf(&self) -> PathBuf {
        self.0.clone()
    }
}

/// Available records for target USGS station.
#[derive(Debug, Default, Clone)]
pub struct StationRecords {
    /// Water discharge records.
    pub discharge_records: Vec<RawRecord>,

    /// Water gage height records.
    pub gage_height_records: Vec<RawRecord>,
}

impl StationRecords {
    pub fn new() -> Self {
        Self::default()
    }

    /// Collect records from target CSV files.
    ///
    /// USGS provides individual CSV files for specific parameter codes. We
    /// mainly care about extracting records from discharge and gage height
    /// CSV files.
    ///
    /// ## Errors
    ///
    /// - Return `Error::InvalidParamterCode` if caller submits a CSV file that
    ///   does not use the correct parameter code for discharge or gage height
    ///   data.
    /// - Return `Error::UnknownParameterCode` if caller submits a CSV file that
    ///   uses a parameter code we do not use.
    /// - Return `Error::CsvError` if CSV file parsing fails for some reason.
    pub fn from_csv_paths(
        discharge_csv_path: DischargeCsvPath,
        gage_height_csv_path: GageHeightCsvPath,
    ) -> Result<Self, Error> {
        let discharge_records = csv_to_raw_records(discharge_csv_path.as_path())?;
        let gage_height_records = csv_to_raw_records(gage_height_csv_path.as_path())?;

        // INVARIANT: Records from discharge CSV path use correct parameter code.
        for record in &discharge_records {
            if record.parameter_code != ParameterCode::Discharge {
                return Err(Error::InvalidParamterCode {
                    path: discharge_csv_path.to_path_buf(),
                    expect: ParameterCode::Discharge,
                    result: record.parameter_code.clone(),
                });
            }
        }

        // INVARIANT: Records from gage height CSV path use correct parameter code.
        for record in &gage_height_records {
            if record.parameter_code != ParameterCode::GageHeight {
                return Err(Error::InvalidParamterCode {
                    path: gage_height_csv_path.to_path_buf(),
                    expect: ParameterCode::GageHeight,
                    result: record.parameter_code.clone(),
                });
            }
        }

        Ok(Self {
            discharge_records,
            gage_height_records,
        })
    }
}

fn csv_to_raw_records(path: &Path) -> Result<Vec<RawRecord>, Error> {
    let mut csv = csv::Reader::from_path(path)?;

    let records = csv
        .deserialize()
        .flat_map(|result| {
            let record: RawRecord = result?;
            Ok::<RawRecord, Error>(record)
        })
        .collect::<Vec<_>>();

    Ok(records)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CSV file {path} has parameter code {result}, expect {expect}")]
    InvalidParamterCode {
        path: PathBuf,
        expect: ParameterCode,
        result: ParameterCode,
    },

    #[error("Unknown parameter code {code}")]
    UnknownParameterCode { code: String },

    #[error(transparent)]
    ChronoError(#[from] chrono::ParseError),

    #[error(transparent)]
    CsvError(#[from] csv::Error),
}
