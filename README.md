# video-quality-filter

Пайплайн metadata для видео-датасетов перед обучением мультимодальных моделей. Он сначала снимает признаки, затем отдельно применяет политику фильтрации. Исходные ролики только читаются: кадры и фрагмент звука живут во временном каталоге и удаляются после каждого файла. В объектное хранилище, если оно включено, уходят только JSONL и Parquet.

## Требования

- Python 3.10+
- `ffmpeg` и `ffprobe` в `PATH`

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Сначала обновите pip. В pip 23.0 и старше установка с большим деревом зависимостей падает с `AssertionError` в `get_topological_weights` уже после сборки колеса.

Дополнительно:

```bash
pip install -e ".[vlm]"          # Qwen-VL: визуальные метрики и video-VLM
pip install -e ".[internvideo]"  # InternVideo2: video/text embeddings
pip install -e ".[storage]"      # выгрузка metadata в S3 или MinIO
pip install -e ".[dvc]"          # DVC: версии видео и отчётов вне git
```

## Запуск

```bash
video-quality-filter run \
  --input ./data/raw \
  --config config/default.yaml \
  --output-dir ./reports
```

По умолчанию считаются технические поля, визуальные метрики по кадрам, сырые аудиопризнаки и дедуп по гистограммам кадров. Qwen3-VL-8B и InternVideo2 выключены: это отдельные проходы, их веса скачиваются локально при первом запуске `config/qwen_vl.yaml`. На CPU 8B заметно тяжелее прежней 3B: в bf16 модели нужно около 16 ГБ, в float32 — больше.

```bash
video-quality-filter run --input ./data/raw --config config/qwen_vl.yaml --output-dir ./reports
```

Код выхода `0` — прогон завершился. Код `2` — нет каталога, сломан конфиг или нет FFmpeg.

## Что считается

Технические поля: `duration`, `fps_original`, `fps_processed`, `width`, `height`, `codec`, `audio_present`, `audio_duration`, `av_duration_diff`, `technical_ok`.

`fps_processed` — частота после нормализации. Если исходный FPS выше `target_fps` (24), в metadata пишется 24: такой ролик можно проредить, и высокий FPS сам по себе не причина отклонения. Более низкий FPS остаётся как есть; слишком низкий отклоняется. Апсемплинг не делается.

Визуальные метрики считаются по равномерно взятым кадрам и усредняются: `aesthetic_score`, `watermark_probability`, `text_area_ratio`. Эвристика — это прокси без скачивания модели. `visual.backend: qwen_vl` заменяет её отдельным проходом Qwen-VL.

Video-VLM, если `vlm.backend: qwen_vl`, делает два прохода разными моделями. Caption пишет `qwen.model_id` (по умолчанию Qwen3-VL-8B). Судья `vlm.judge_model_id` (`lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`) пишет свой `judge_caption` по тем же кадрам и ставит оценки `semantic_consistency`, `temporal_coverage`, `completeness`, `hallucination` чужому caption. В блоке `vlm` поле `judge_model` — id судьи. Для судьи нужен `transformers>=5.7`. В bf16 это ещё около 16 ГБ поверх Qwen.

Video-text alignment использует InternVideo2. `video_text.cosine_similarity` — косинус video embedding и text embedding caption. Это сходство в диапазоне примерно от -1 до 1, не вероятность. Softmax от `100 * cosine`, который встречается в демо retrieval, сюда не записывается.

`video_text.replace_words` подменяет целые слова только в строке, которую видит InternVideo2. Caption Qwen в metadata не меняется. Фильтр по-прежнему смотрит на исходный `cosine_similarity`. Рядом пишутся `caption_replaced` и `cosine_similarity_replaced`.

```yaml
video_text:
  replace_words:
    man: hand
```

`OpenGVLab/InternVideo2-Stage2_1B-224p-f4` на Hugging Face — только файл `InternVideo2-stage2_1b-224p-f4.pt`, без `config.json`, поэтому `AutoModel` его не открывает. Загрузчик берёт код Stage2 из `OpenGVLab/InternVideo2-Stage2_6B` (не веса 6B) и этот `.pt`, а текстовый энкодер — `bert-large-uncased`. Репозиторий весов закрыт: примите условия на странице модели и выполните `huggingface-cli login`.

Аудио: `silence_ratio`, `rms`, `clipping_ratio`. Рядом лежит агрегат `audio_quality`, но политика смотрит на сырые признаки.

Дедуп по video embedding (гистограмма кадров или InternVideo2): `cluster_id`, `nearest_video_id`, `similarity`, `is_near_duplicate`. Файлы не удаляются. При `reject_near_duplicates: true` в батче остаётся один канонический id компонента, остальные получают причину `near_duplicate`.

## Фильтрация

Блок `filtering` отделён от признаков:

- `status`: `accepted` или `rejected`
- `reasons`: список кодов

Общего score нет. Порог `null` выключает проверку. Пустое значение метрики не отклоняет ролик, пока не включён соответствующий `require_visual`, `require_vlm` или `require_video_text`.

Технические причины: `unreadable`, `no_video_stream`, `duration_too_short`, `duration_too_long`, `duration_unknown`, `resolution_too_small`, `fps_too_low`, `fps_unknown`, `codec_invalid`, `codec_not_allowed`, `audio_required_but_missing`, `av_duration_mismatch`.

Остальные: `aesthetic_score_below_threshold`, `watermark_probability_above_threshold`, `text_area_ratio_above_threshold`, `semantic_consistency_below_threshold`, `temporal_coverage_below_threshold`, `completeness_below_threshold`, `hallucination_above_threshold`, `video_text_cosine_below_threshold`, `silence_ratio_above_threshold`, `rms_below_threshold`, `clipping_ratio_above_threshold`, `near_duplicate`, `frame_extraction_failed`, `visual_extraction_failed`, `vlm_failed`, `video_text_failed`, `audio_analysis_failed`, `visual_required_but_missing`, `vlm_required_but_missing`, `video_text_required_but_missing`, `processing_error`.

Имена кодеков такие, как у ffprobe (`h264`, `hevc`, `vp9`, `av1`). `h265` принимается как `hevc`.

## Metadata

Каждый запуск заново пишет `results.jsonl` и `results.parquet`.

```json
{
  "video_id": "000123",
  "source": "000123.mp4",
  "technical": {
    "duration": 12.4,
    "fps_original": 60,
    "fps_processed": 24,
    "width": 1920,
    "height": 1080,
    "codec": "h264",
    "audio_present": true,
    "audio_duration": 12.2,
    "av_duration_diff": 0.2,
    "technical_ok": true
  },
  "visual": {
    "aesthetic_score": 0.82,
    "watermark_probability": 0.03,
    "text_area_ratio": 0.04
  },
  "vlm": {
    "caption": "A man opens a car door and gets into the vehicle.",
    "judge_model": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
    "judge_caption": "A hand holds a photo of a car door.",
    "semantic_consistency": 0.93,
    "temporal_coverage": 0.84,
    "completeness": 0.81,
    "hallucination": 0.05
  },
  "video_text": {
    "model": "InternVideo2",
    "cosine_similarity": 0.78
  },
  "audio": {
    "silence_ratio": 0.04,
    "rms": 0.18,
    "clipping_ratio": 0.002,
    "audio_quality": 0.91
  },
  "deduplication": {
    "cluster_id": 152,
    "nearest_video_id": "000981",
    "similarity": 0.97,
    "is_near_duplicate": true
  },
  "filtering": {
    "status": "accepted",
    "reasons": []
  }
}
```

`video_id` — имя файла без расширения. `source` — путь относительно входного каталога. В блоке `audio` дополнительно есть `audio_quality`: это только агрегат трёх сырых признаков.

S3/MinIO:

```yaml
storage:
  enabled: true
  bucket: datasets
  prefix: metadata
  endpoint_url: http://localhost:9000
  region: us-east-1
```

Ключи и секреты берутся из обычного окружения boto3 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

## DVC

Исходные ролики и отчёты не коммитятся в git. Их содержимое хранит DVC, в репозитории остаются только указатели `data/raw.dvc` и `dvc.lock`. Фильтр по-прежнему не удаляет и не перекодирует файлы в `data/raw`.

- `data/raw` — видео, которые читает пайплайн.
- `data/reports` — `results.jsonl` и `results.parquet` после `dvc repro`.

По умолчанию remote `local`: каталог `dvc-storage` рядом с репозиторием. Рядом настроен remote `s3` (`s3://datasets/video-quality-filter`) без ключей. Секреты и endpoint MinIO пишутся только в `.dvc/config.local`:

```bash
dvc remote modify --local s3 endpointurl http://127.0.0.1:9000
dvc remote modify --local s3 access_key_id "$AWS_ACCESS_KEY_ID"
dvc remote modify --local s3 secret_access_key "$AWS_SECRET_ACCESS_KEY"
dvc remote default --local s3
```

Шаблон этих полей лежит в `.dvc/config.local.example`.

```bash
# новые или изменённые ролики
dvc add data/raw
dvc push

# прогон и версия отчётов
dvc repro
dvc push
```

На другой машине: `dvc pull`. Обычный запуск CLI по-прежнему может писать в `./reports`; этот каталог в DVC не входит.

## Тесты

```bash
pytest
```

## Ограничения

Эвристические `aesthetic_score`, `watermark_probability` и `text_area_ratio` — дешёвый прокси. Для датасета их лучше заменить проходом Qwen-VL и откалибровать пороги. InternVideo2 не апсемплит низкий FPS и не перекодирует файл: `fps_processed` описывает целевую частоту, а не новый ролик на диске.
