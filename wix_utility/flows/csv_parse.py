"""CSV parsing flow."""

from __future__ import annotations

from wix_utility.core.config import WixConfig
from wix_utility.flows.base import FlowResult, WixFlow
from wix_utility.io.csv_records import parse_csv_records, write_json_records


class ParseCsvFlow(WixFlow):
    flow_name = "parse-csv"

    def run(self, config: WixConfig) -> FlowResult:
        if config.input_csv is None:
            self.logger.error("WIX_INPUT_CSV is not set.")
            return FlowResult(exit_code=1, flow_name=self.flow_name, message="missing WIX_INPUT_CSV")
        if not config.input_csv.is_file():
            self.logger.error("CSV file does not exist: %s", config.input_csv)
            return FlowResult(exit_code=1, flow_name=self.flow_name, message="CSV file not found")

        records = parse_csv_records(config.input_csv, delimiter=config.csv_delimiter)
        self.logger.info("Parsed CSV records=%s path=%s", len(records), config.input_csv)

        if config.csv_output_json is not None:
            write_json_records(records, config.csv_output_json)
            self.logger.info("Wrote JSON records to %s", config.csv_output_json)

        return FlowResult(exit_code=0, flow_name=self.flow_name, records_loaded=len(records))
