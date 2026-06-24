#!/usr/bin/env python3
"""Watch the input folder and process PDFs as they arrive."""

import logging
import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from extract import INPUT_DIR, OUTPUT_DIR, load_model_and_tokenizer, process_pdf

POLL_INTERVAL = float(os.environ.get("WATCH_POLL_INTERVAL", "2"))
STABLE_CHECKS = int(os.environ.get("WATCH_STABLE_CHECKS", "2"))
PROCESS_EXISTING = os.environ.get("PROCESS_EXISTING", "true").lower() in {
    "1",
    "true",
    "yes",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"


def wait_for_stable_file(path: Path) -> bool:
    """Wait until the file size stops changing (copy/upload finished)."""
    if not path.exists():
        return False

    last_size = -1
    stable_count = 0

    while stable_count < STABLE_CHECKS:
        if not path.exists():
            return False

        size = path.stat().st_size
        if size > 0 and size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size

        time.sleep(POLL_INTERVAL)

    return True


class PdfProcessor:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._in_progress: set[str] = set()

    def handle(self, pdf_path: Path) -> None:
        key = str(pdf_path.resolve())
        if key in self._in_progress:
            return

        if not is_pdf(pdf_path):
            return

        self._in_progress.add(key)
        try:
            if not wait_for_stable_file(pdf_path):
                logger.warning("File disappeared before processing: %s", pdf_path.name)
                return

            process_pdf(pdf_path, self.model, self.tokenizer, output_dir=OUTPUT_DIR)
            logger.info("Finished processing %s", pdf_path.name)
        except Exception:
            logger.exception("Failed to process %s", pdf_path.name)
        finally:
            self._in_progress.discard(key)


class InputFolderHandler(FileSystemEventHandler):
    def __init__(self, processor: PdfProcessor) -> None:
        self.processor = processor

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.processor.handle(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.processor.handle(Path(event.dest_path))


def process_existing_files(processor: PdfProcessor) -> None:
    for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
        processor.handle(pdf_path)


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Input folder:  %s", INPUT_DIR)
    logger.info("Output folder: %s", OUTPUT_DIR)

    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception:
        logger.exception("Failed to load model")
        return 1

    processor = PdfProcessor(model, tokenizer)

    if PROCESS_EXISTING:
        process_existing_files(processor)

    handler = InputFolderHandler(processor)
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()

    logger.info("Watching for new PDFs in %s", INPUT_DIR)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()

    observer.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
