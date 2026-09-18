"""Reports retain evidence and uncertainty; aggregation does not invent new events."""

from __future__ import annotations

import csv
import html

from .media import write_json

LABELS = {
    "hard_cut": "Резкая склейка",
    "crop_jump": "Скачок крупности",
    "gradual_transition": "Плавный переход",
    "overlay": "Наложение/графика",
    "camera_motion": "Движение камеры",
    "no_change": "Монтажная граница не подтверждена",
    "uncertain": "Требует просмотра",
    "speed_change": "Предполагаемая смена скорости",
}
RECREATE = {
    "hard_cut": "Соединить два плана прямой склейкой в указанной позиции.",
    "crop_jump": "Разрезать клип и ступенчато изменить кадрирование/масштаб на втором фрагменте.",
    "gradual_transition": (
        "Повторить видимую смену изображения ключевыми кадрами; "
        "длительность уточнить по соседним кадрам."
    ),
    "overlay": "Разместить графику отдельным слоем; повторить положение и анимацию по кадрам.",
    "camera_motion": (
        "Повторить движение при съёмке или ключевыми кадрами; способ по видео не установлен."
    ),
    "no_change": "Дополнительный монтажный разрез здесь не требуется по согласованному вердикту.",
    "uncertain": "Сравнить соседние кадры вручную; способ воспроизведения пока не установлен.",
}


def stamp(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def render(plan, result, output):
    write_json(output / "result.json", result)
    usage = result["usage"]
    heading = [
        "# Монтажный разбор",
        "",
        f"Модель: {result['provider']['MONTAGE_MODEL']}; обработка: **{result['status']}**; "
        f"проверка границ: **{result['quality_status']}**.",
        f"Кадры: {result['analyzed_frames']}/{result['source_frames']}; "
        f"окна: {result['completed_windows']}/{result['total_windows']}; "
        f"кандидаты: {len(result['boundaries'])}.",
        f"Попытки API: {usage['attempts']}; известные входные/выходные токены: "
        f"{usage['input_tokens']}/{usage['output_tokens']}; "
        f"попытки с неизвестным расходом: {usage['unknown_usage_attempts']}.",
        "Таймкоды вычислены из PTS входного файла и явно указанного смещения. "
        "Индексы кадров относятся к входному файлу.",
        "Денежный счёт зависит от тарифа провайдера; неизвестный расход не считается нулевым.",
        "",
    ]
    lines = heading + [
        "## Длительности планов",
        "",
        "Разбиение по согласованным резким границам; плавные переходы описаны отдельно.",
        "",
        "| План | Начало | Конец | Длительность |",
        "|---|---|---|---|",
    ]
    for shot in result["shots"]:
        lines.append(
            f"| {shot['id']} | {stamp(shot['start'])} | {stamp(shot['end'])} | "
            f"{shot['duration']:.3f} с |"
        )
    with (output / "shots.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "start", "end", "duration", "first_frame_index", "last_frame_index"],
        )
        writer.writeheader()
        writer.writerows(result["shots"])
    lines += ["", "## Монтажный лист", ""]
    cards = []
    rows = []
    escape = html.escape
    for boundary in result["boundaries"]:
        c = boundary
        label = LABELS[c["kind"]]
        line = f"{stamp(c['time'])} — {label} ({c['status']}, {c['id']})"
        lines += [
            f"### {line}",
            "",
            c["description"],
            "",
            f"До: {c['before']} После: {c['after']}",
            "",
            RECREATE[c["kind"]],
            "",
            f"[Кадр до]({c['left']['path']}) · [Кадр после]({c['right']['path']})",
            "",
        ]
        cards.append(
            f"<article><h3>{escape(line)}</h3><p>{escape(c['description'])}</p>"
            f"<p>{escape(RECREATE[c['kind']])}</p><div class='pair'>"
            + "".join(
                f"<figure><img loading='lazy' src='{escape(c[s]['path'], quote=True)}'>"
                f"<figcaption>{s}: {stamp(c[s]['time'])}, "
                f"frame {c[s]['index']}</figcaption></figure>"
                for s in ("left", "right")
            )
            + "</div><details><summary>Вердикты проверок</summary><ul>"
            + "".join(
                "<li>" + escape(v["stage"] + ": " + v["kind"] + " — " + v["description"]) + "</li>"
                for v in c["votes"]
            )
            + "</ul></details></article>"
        )
        rows.append(
            {
                "time_seconds": c["time"],
                "timecode": stamp(c["time"]),
                "frame_index": c["index"],
                "kind": c["kind"],
                "status": c["status"],
                "description": c["description"],
                "before_image": c["left"]["path"],
                "after_image": c["right"]["path"],
            }
        )
    with (output / "montage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_seconds",
                "timecode",
                "frame_index",
                "kind",
                "status",
                "description",
                "before_image",
                "after_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    lines += [
        "## Вставки, движение, анимация",
        "",
        "Наблюдения основного прохода; интервалы соседних окон могут перекрываться.",
        "",
    ]
    for e in sorted(result["events"], key=lambda x: (x["start"], x["end"])):
        lines += [
            f"- {stamp(e['start'])}–{stamp(e['end'])}: {e['description']} "
            f"({e['confidence']}, {e['window']})."
        ]
    lines += ["", "## Как повторить по участкам", ""]
    for w in result["windows"]:
        lines += [f"### {stamp(w['start'])}–{stamp(w['end'])}", "", w["summary"], ""]
        lines += [f"- {step}" for step in w.get("recreation_steps", [])]
        lines += [f"- Неопределённость: {u}" for u in w["uncertainties"]] + [""]
    lines += ["## Ограничения", ""] + [f"- {s}" for s in result["limitations"]]
    if result["failures"]:
        lines += ["", "## Незавершённые операции", ""] + [
            f"- {r['task']}: {r['reason']}" for r in result["failures"]
        ]
    markdown = "\n".join(lines) + "\n"
    (output / "report.md").write_text(markdown, encoding="utf-8")
    document = """<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Монтажный разбор</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;
background:#fafafa;color:#181818}.pair{display:flex;gap:12px}
figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}
article{padding:20px;margin:20px 0;background:white;border:1px solid #ddd;border-radius:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}
@media(max-width:650px){.pair{display:block}}</style>"""
    document += (
        "<h1>Монтажный разбор</h1><pre>"
        + escape("\n".join(heading[2:]))
        + "</pre>"
        + "".join(cards)
    )
    document += "<h2>Полный текст</h2><pre>" + escape(markdown) + "</pre></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
