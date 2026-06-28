"""Media upload flow boilerplate."""

from __future__ import annotations

from wix_utility.core.config import FLOW_MEDIA_UPLOAD, WixConfig
from wix_utility.flows.base import FlowResult, WixFlow


class MediaUploadFlow(WixFlow):
    flow_name = FLOW_MEDIA_UPLOAD

    def run(self, config: WixConfig) -> FlowResult:
        if config.image_dir is None:
            self.logger.warning("WIX_IMAGE_DIR is not set; media upload flow has no images to inspect.")
            return FlowResult(exit_code=0, flow_name=self.flow_name, message="missing WIX_IMAGE_DIR")
        if not config.image_dir.is_dir():
            self.logger.error("Image directory does not exist: %s", config.image_dir)
            return FlowResult(exit_code=1, flow_name=self.flow_name, message="image directory not found")

        files = sorted(path for path in config.image_dir.iterdir() if path.is_file())
        self.logger.info("Media upload boilerplate found files=%s image_dir=%s", len(files), config.image_dir)
        self.logger.info("Next step will request Wix upload URLs and attach media to products.")
        return FlowResult(exit_code=0, flow_name=self.flow_name, records_loaded=len(files))
