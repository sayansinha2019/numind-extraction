#!/usr/bin/env python3
"""Extract structured JSON from PDFs using numind/NuExtract3."""

import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from excel_export import write_excel_from_extraction

MODEL_ID = os.environ.get("MODEL_ID", "numind/NuExtract3")
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get("INPUT_DIR", BASE_DIR / "input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "output"))
TEMPLATE_INSTRUCTION = (
    "Generate a JSON template for this document. "
    "Include every field, table, and nested section visible across all pages."
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
PDF_DPI = int(os.environ.get("PDF_DPI", "400"))
TEMPLATE_BATCH_SIZE = int(os.environ.get("TEMPLATE_BATCH_SIZE", "0"))
EXTRACTION_BATCH_SIZE = int(os.environ.get("EXTRACTION_BATCH_SIZE", "6"))
GPU_OOM_RETRIES = int(os.environ.get("GPU_OOM_RETRIES", "2"))
AGGRESSIVE_GPU_CLEANUP = os.environ.get("AGGRESSIVE_GPU_CLEANUP", "false").lower() in {
    "1",
    "true",
    "yes",
}


def effective_batch_size(batch_size: int, page_count: int) -> int:
    """Treat 0 as 'all pages in one batch' for high-VRAM processing."""
    if batch_size <= 0:
        return max(page_count, 1)
    return batch_size


def load_model_and_tokenizer():
    """Load NuExtract3 model and tokenizer (via AutoProcessor) onto GPU."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Loading model {MODEL_ID}...")
    print(
        f"Quality profile: PDF_DPI={PDF_DPI}, "
        f"template_batch={TEMPLATE_BATCH_SIZE or 'all'}, "
        f"extraction_batch={EXTRACTION_BATCH_SIZE or 'all'}, "
        f"max_tokens={MAX_NEW_TOKENS}"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print("Model loaded successfully.")
    return model, tokenizer


def clear_gpu_cache() -> None:
    """Release Python and GPU memory between inference calls."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def maybe_clear_gpu_cache(*, force: bool = False) -> None:
    """Clear GPU memory when running in conservative mode or on demand."""
    if force or AGGRESSIVE_GPU_CLEANUP:
        clear_gpu_cache()


def pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    info = pdfinfo_from_path(str(pdf_path))
    pages = info.get("Pages")
    if not pages:
        raise ValueError(f"No pages found in {pdf_path.name}")
    return int(pages)


def pdf_page_to_image(pdf_path: Path, page_number: int) -> Image.Image:
    """Convert a single PDF page to a high-quality PIL RGB image."""
    pages = convert_from_path(
        str(pdf_path),
        dpi=PDF_DPI,
        first_page=page_number,
        last_page=page_number,
        fmt="png",
        grayscale=False,
        use_pdftocairo=True,
        thread_count=1,
    )
    if not pages:
        raise ValueError(f"Could not render page {page_number} of {pdf_path.name}")
    return pages[0].convert("RGB")


def build_image_message(images: Image.Image | list[Image.Image]) -> list[dict]:
    """Build a chat message array containing one or more document images."""
    if isinstance(images, Image.Image):
        image_list = [images]
    else:
        image_list = images

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                }
                for image in image_list
            ],
        }
    ]


def apply_chat_template_and_generate(
    model,
    tokenizer,
    messages: list[dict],
    *,
    task: str,
    instruction: str | None = None,
    template: str | None = None,
) -> str:
    """Apply NuExtract chat template and run model inference."""
    chat_template_kwargs: dict = {"enable_thinking": False}

    if task == "template_generation":
        chat_template_kwargs["mode"] = "template-generation"
        if instruction:
            chat_template_kwargs["instructions"] = instruction
    elif task == "extraction":
        if not template:
            raise ValueError("template is required for extraction task")
        chat_template_kwargs["template"] = template
    else:
        raise ValueError(f"Unsupported task: {task}")

    inputs = None
    generated_ids = None

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        ).to(model.device)

        for attempt in range(GPU_OOM_RETRIES + 1):
            try:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=0.0,
                    )
                break
            except torch.cuda.OutOfMemoryError:
                if attempt >= GPU_OOM_RETRIES:
                    raise
                print(
                    f"GPU out of memory during {task} "
                    f"(attempt {attempt + 1}/{GPU_OOM_RETRIES + 1}) — retrying...",
                    file=sys.stderr,
                )
                clear_gpu_cache()
                time.sleep(3)

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        return tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    finally:
        if generated_ids is not None:
            del generated_ids
        if inputs is not None:
            del inputs
        maybe_clear_gpu_cache()


def parse_json_output(text: str) -> dict:
    """Parse JSON from model output, handling markdown fences and extra text."""
    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def is_empty(value: object) -> bool:
    """Return True when a value carries no extracted information."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def merge_strings(existing: str, new: str) -> str:
    """Merge string values that may continue across pages."""
    left = existing.strip()
    right = new.strip()
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    if left in right:
        return right
    if right in left:
        return left
    return f"{left} {right}"


def merge_lists(existing: list, new: list) -> list:
    """Merge list values, preserving rows that appear on later pages."""
    if not existing:
        return list(new)
    if not new:
        return list(existing)

    if all(isinstance(item, dict) for item in existing + new):
        return list(existing) + list(new)

    merged = list(existing)
    for item in new:
        if item not in merged:
            merged.append(item)
    return merged


def merge_values(existing: object, new: object) -> object:
    """Merge two extracted values for the same key across pages."""
    if is_empty(new):
        return existing
    if is_empty(existing):
        return new

    if isinstance(existing, dict) and isinstance(new, dict):
        return deep_merge_data(existing, new)
    if isinstance(existing, list) and isinstance(new, list):
        return merge_lists(existing, new)
    if isinstance(existing, str) and isinstance(new, str):
        return merge_strings(existing, new)

    if isinstance(existing, (int, float)) and isinstance(new, (int, float)):
        return new

    return merge_strings(str(existing), str(new))


def deep_merge_template(existing: dict, new: dict) -> dict:
    """Merge JSON templates so every field from every page is represented."""
    merged = dict(existing)
    for key, new_value in new.items():
        if key not in merged:
            merged[key] = new_value
        elif isinstance(merged[key], dict) and isinstance(new_value, dict):
            merged[key] = deep_merge_template(merged[key], new_value)
        elif isinstance(new_value, dict):
            merged[key] = new_value
    return merged


def deep_merge_data(existing: dict, new: dict) -> dict:
    """Merge extracted page data into one document-level JSON object."""
    merged = dict(existing)
    for key, new_value in new.items():
        if key in merged:
            merged[key] = merge_values(merged[key], new_value)
        else:
            merged[key] = new_value
    return merged


def generate_template(
    model, tokenizer, images: Image.Image | list[Image.Image]
) -> str:
    """Generate a JSON template from one or more page images."""
    messages = build_image_message(images)
    template_text = apply_chat_template_and_generate(
        model,
        tokenizer,
        messages,
        task="template_generation",
        instruction=TEMPLATE_INSTRUCTION,
    )
    return template_text


def extract_data(
    model, tokenizer, images: Image.Image | list[Image.Image], template_text: str
) -> str:
    """Extract structured data from one or more page images."""
    messages = build_image_message(images)
    return apply_chat_template_and_generate(
        model,
        tokenizer,
        messages,
        task="extraction",
        template=template_text,
    )


def load_page_images(pdf_path: Path, page_numbers: list[int]) -> list[Image.Image]:
    """Load a small batch of page images and keep GPU memory usage low."""
    return [pdf_page_to_image(pdf_path, page_number) for page_number in page_numbers]


def close_images(images: list[Image.Image]) -> None:
    """Close PIL images after a batch is processed."""
    for image in images:
        image.close()


def build_document_template(
    model, tokenizer, pdf_path: Path, page_count: int
) -> tuple[str, list[dict]]:
    """Build a unified template by scanning pages in configurable batches."""
    merged_template: dict = {}
    errors: list[dict] = []
    batch_size = effective_batch_size(TEMPLATE_BATCH_SIZE, page_count)

    for batch_start in range(1, page_count + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, page_count)
        page_numbers = list(range(batch_start, batch_end + 1))
        images = load_page_images(pdf_path, page_numbers)

        try:
            print(
                f"Building template from page(s) "
                f"{page_numbers[0]}-{page_numbers[-1]}/{page_count}..."
            )
            template_text = generate_template(
                model,
                tokenizer,
                images[0] if len(images) == 1 else images,
            )
            merged_template = deep_merge_template(
                merged_template,
                parse_json_output(template_text),
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "template",
                    "pages": page_numbers,
                    "error": str(exc),
                }
            )
            print(
                f"Template generation failed for pages {page_numbers}: {exc}",
                file=sys.stderr,
            )
        finally:
            close_images(images)
            maybe_clear_gpu_cache()

    if not merged_template:
        raise ValueError(f"Could not build a template for {pdf_path.name}")

    return json.dumps(merged_template, indent=2, ensure_ascii=False), errors


def extract_document_pages(
    model,
    tokenizer,
    pdf_path: Path,
    page_count: int,
    template_text: str,
) -> tuple[list[dict], dict, list[dict]]:
    """Extract pages in batches and merge values across the full document."""
    pages_data: list[dict] = []
    merged_data: dict = {}
    errors: list[dict] = []
    batch_size = effective_batch_size(EXTRACTION_BATCH_SIZE, page_count)

    for batch_start in range(1, page_count + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, page_count)
        page_numbers = list(range(batch_start, batch_end + 1))
        images = load_page_images(pdf_path, page_numbers)
        page_entry = {
            "pages": page_numbers,
            "status": "ok",
            "data": {},
        }

        try:
            print(
                f"Extracting page(s) {page_numbers[0]}-{page_numbers[-1]}/{page_count}..."
            )
            extracted_text = extract_data(
                model,
                tokenizer,
                images[0] if len(images) == 1 else images,
                template_text,
            )
            batch_data = parse_json_output(extracted_text)
            page_entry["data"] = batch_data
            merged_data = deep_merge_data(merged_data, batch_data)
        except Exception as exc:
            page_entry["status"] = "error"
            page_entry["error"] = str(exc)
            errors.append(
                {
                    "stage": "extraction",
                    "pages": page_numbers,
                    "error": str(exc),
                }
            )
            print(
                f"Extraction failed for pages {page_numbers}: {exc}",
                file=sys.stderr,
            )
        finally:
            close_images(images)
            maybe_clear_gpu_cache()

        pages_data.append(page_entry)

    return pages_data, merged_data, errors


def resolve_output_folder(pdf_path: Path, output_dir: Path) -> Path:
    """Return a unique folder under output_dir for this PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / pdf_path.stem
    if not candidate.exists():
        return candidate

    for index in range(1, 1000):
        numbered = output_dir / f"{pdf_path.stem}_{index}"
        if not numbered.exists():
            return numbered

    raise RuntimeError(f"Could not allocate output folder for {pdf_path.name}")


def save_results(pdf_path: Path, data: dict, output_dir: Path) -> Path:
    """Save PDF, JSON, and Excel into a dedicated folder under output_dir."""
    folder = resolve_output_folder(pdf_path, output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    json_path = folder / f"{pdf_path.stem}.json"
    excel_path = folder / f"{pdf_path.stem}.xlsx"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    write_excel_from_extraction(
        data,
        excel_path,
        source_name=pdf_path.name,
    )

    shutil.move(str(pdf_path), str(folder / pdf_path.name))
    return folder


def process_pdf(pdf_path: Path, model, tokenizer, *, output_dir: Path | None = None) -> Path:
    """Process a multi-page PDF with GPU-safe batching and merged output."""
    target_output_dir = output_dir or OUTPUT_DIR
    print(f"Processing {pdf_path.name}...")

    page_count = pdf_page_count(pdf_path)
    print(f"Found {page_count} page(s)...")

    template_text, template_errors = build_document_template(
        model,
        tokenizer,
        pdf_path,
        page_count,
    )
    pages_data, merged_data, extraction_errors = extract_document_pages(
        model,
        tokenizer,
        pdf_path,
        page_count,
        template_text,
    )

    extracted_data = {
        "page_count": page_count,
        "data": merged_data,
        "pages": pages_data,
        "errors": template_errors + extraction_errors,
    }

    result_folder = save_results(pdf_path, extracted_data, target_output_dir)
    print(f"Saved results to {result_folder}")
    clear_gpu_cache()
    return result_folder


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}. Place PDFs in the input/ folder.")
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    succeeded = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path, model, tokenizer)
            succeeded += 1
        except Exception as exc:
            failed += 1
            print(f"Skipping {pdf_path.name}: {exc}", file=sys.stderr)
            clear_gpu_cache()

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
