import logging
import warnings

import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("astra_pipeline")


def get_ram_info() -> str:
    rss_gb = 0.0
    avail_gb = 0.0
    total_gb = 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / (1024 * 1024)
                    break
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_gb = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return (
        f"Process RAM: {rss_gb:.2f} GB | System Avail: {avail_gb:.1f}/{total_gb:.1f} GB"
    )


def log_step(msg: str) -> None:
    logger.info(f"[{get_ram_info()}] {msg}")
