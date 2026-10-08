import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from PIL import Image

from video_quality_filter.config import FramesConfig, QwenConfig
from video_quality_filter.extract import extract_frames
from video_quality_filter.probe import probe_video
from video_quality_filter.visual_qwen import load_qwen


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("🎥 Введите полный путь к видеофайлу: ").strip()

    video_file = Path(video_path).expanduser()
    if not video_file.is_file():
        print(f"❌ Файл не найден по пути: {video_path}")
        return

    frames_config = FramesConfig()
    probed = probe_video(video_file, timeout=frames_config.timeout_sec)
    if probed.probe_error or not probed.duration_sec:
        print(f"❌ Не удалось прочитать видео: {probed.probe_error or 'нет длительности'}")
        return

    print("🎞️ Снимаю кадры...")
    with tempfile.TemporaryDirectory() as temporary:
        extracted, warnings = extract_frames(
            video_file,
            probed.duration_sec,
            frames_config,
            Path(temporary),
        )
        images = [Image.fromarray(frame.image).convert("RGB") for frame in extracted]
    for warning in warnings:
        print(f"⚠️ {warning}")
    if not images:
        print("❌ Кадры не сняты")
        return

    config = QwenConfig(max_new_tokens=512)
    print(f"🚀 Загрузка {config.model_id}...")
    model, processor = load_qwen(config)

    chat_history = []
    is_first_turn = True
    print(f"🎞️ Кадров: {len(images)}, как в пайплайне (frames.count={frames_config.count}).")
    print("\n✨ Модель готова к работе! Пишите свои вопросы. Для выхода введите 'exit' или 'quit'.")
    print("-" * 60)

    try:
        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Чат завершен.")
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Чат завершен.")
                break

            content = [{"type": "text", "text": user_input}]
            if is_first_turn:
                content = [{"type": "image", "image": image} for image in images] + content
                is_first_turn = False
            chat_history.append({"role": "user", "content": content})

            print("🤖 Qwen думает...")
            output_text = _answer(model, processor, chat_history, config.max_new_tokens)
            print(f"🤖 Qwen: {output_text}")
            chat_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]}
            )
    finally:
        chat_history.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _answer(model, processor, messages, max_new_tokens: int) -> str:
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except (TypeError, ValueError):
        from video_quality_filter.visual_qwen import _prepare_inputs_legacy

        inputs = _prepare_inputs_legacy(processor, messages)

    device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, input_len:] if generated.shape[-1] > input_len else generated
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    main()
