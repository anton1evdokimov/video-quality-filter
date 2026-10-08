import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 512
# Кадр до 480×480, весь ролик — не больше 32 кадров: иначе длинное видео не влезает в память.
FRAME_PIXELS = 480 * 480
MAX_FRAMES = 32


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("🎥 Введите полный путь к видеофайлу: ").strip()

    video_file = Path(video_path).expanduser()
    if not video_file.is_file():
        print(f"❌ Файл не найден по пути: {video_path}")
        return

    print(f"🚀 Загрузка {MODEL_ID}...")
    model = _load_model()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    _limit_video(processor)

    chat_history = []
    is_first_turn = True
    video_uri = video_file.resolve().as_uri()

    print(f"🎞️ Видео: 1 кадр в секунду, не больше {MAX_FRAMES} кадров.")
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

            if is_first_turn:
                user_content = [
                    {
                        "type": "video",
                        "video": video_uri,
                        "fps": 1.0,
                        "max_pixels": FRAME_PIXELS,
                    },
                    {"type": "text", "text": user_input},
                ]
                is_first_turn = False
            else:
                user_content = [{"type": "text", "text": user_input}]

            chat_history.append({"role": "user", "content": user_content})
            print("🤖 Qwen думает...")
            output_text = _answer(model, processor, chat_history)
            print(f"🤖 Qwen: {output_text}")
            # Ответ ассистента — список блоков, как и реплика пользователя.
            chat_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]}
            )
    finally:
        chat_history.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model():
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
    return model.eval()


def _limit_video(processor):
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    if hasattr(video_processor, "size"):
        video_processor.size = {
            "longest_edge": FRAME_PIXELS * MAX_FRAMES,
            "shortest_edge": 224 * 224,
        }
    if hasattr(video_processor, "fps"):
        video_processor.fps = 1.0
    if hasattr(video_processor, "max_frames"):
        video_processor.max_frames = MAX_FRAMES


def _answer(model, processor, chat_history):
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "fps": 1.0,
    }
    try:
        inputs = processor.apply_chat_template(chat_history, **template_kwargs)
    except TypeError:
        template_kwargs.pop("fps")
        inputs = processor.apply_chat_template(chat_history, **template_kwargs)

    if inputs.get("pixel_values_videos") is None:
        raise RuntimeError(
            "Процессор не прочитал видео. Нужен transformers с поддержкой Qwen3-VL (>=4.57)."
        )

    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


if __name__ == "__main__":
    main()
