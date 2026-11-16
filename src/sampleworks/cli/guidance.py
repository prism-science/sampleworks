"""Unified CLI for running diffusion guidance with sampleworks."""

from __future__ import annotations

import sys

from sampleworks.utils.guidance_script_arguments import GuidanceConfig


def main(argv: list[str] | None = None) -> int:
    config = GuidanceConfig.from_cli(argv)

    from loguru import logger

    from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance

    logger.info(f"Running guidance with config: {config}")
    device, model_wrapper = get_model_and_device(
        config.device,
        getattr(config, "model_checkpoint", None),
        config.model_name,
        config=config,
    )
    result = run_guidance(config, model_wrapper, device)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
