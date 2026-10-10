"""Fixed-height worker grid; terminal cells include Korean wide characters."""

import unicodedata


def cell(text, width):
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\x1b", "?")
    result, used = [], 0
    for char in text:
        size = (
            0
            if unicodedata.combining(char)
            else 2
            if unicodedata.east_asian_width(char) in "WF"
            else 1
        )
        if used + size > width:
            break
        result.append(char)
        used += size
    return "".join(result) + " " * (width - used)


def grid(items, width=78):
    """First8 down left, next8 down right. Never16 rows tall."""
    half = max(1, (width - 3) // 2)
    items = list(items)[:16]
    return [
        "│"
        + cell(items[i] if i < len(items) else "", half)
        + "│"
        + cell(items[i + 8] if i + 8 < len(items) else "", half)
        for i in range(min(8, len(items)))
    ]


STATUS = {
    "FLYING": "비행",
    "WAITING_VERIFY_EXPORT": "검증대기",
    "VERIFYING": "검증",
    "ARCHIVING": "압축",
    "VERIFYING_ARCHIVE": "압축검증",
    "FAILED_PRESERVED": "실패",
}


def flight_cards(workers, duration):
    # Old failed cases can share a reused lane; show current lane first, summarize failures separately.
    lanes = {}
    for item in workers:
        lane = item.get("instance")
        previous = lanes.get(lane)
        if previous is None or previous["status"] == "FAILED_PRESERVED":
            lanes[lane] = item
    aliases = {
        "hover": "호버",
        "altitude": "고도",
        "yaw": "회전",
        "horizontal": "수평",
        "waypoint": "경로",
        "low_soc": "저SOC",
        "payload": "탑재",
        "wind": "바람",
        "cold": "저온",
    }
    cards = []
    for lane, item in sorted(lanes.items(), key=lambda pair: str(pair[0]).zfill(2)):
        name = item["scenario"].removesuffix("_normal")
        name = aliases.get(name, name[:8])
        flight = item.get("flight", {})
        elapsed_s = flight.get("observed_simulation_elapsed_s")
        target_s = item.get("target_mission_seconds")
        elapsed = duration(elapsed_s)[3:] if elapsed_s is not None else "--:--"
        target = duration(target_s)[3:] if target_s is not None else "--:--"
        status = STATUS.get(item["status"], item["status"][:8])
        cards.append(
            f"{str(lane):>2} {name} {status} {elapsed}/{target} {item.get('attempt', 1)}/{item.get('attempts_max', 1)}"
        )
    return cards


def screen(text, columns, rows):
    """Drop secondary explanations before clipping; preserve header, workers, error and footer."""
    lines = text.splitlines()
    optional = (
        "흐름",
        "보관   ",
        "전처리   ",
        "수집 완료",
        "단계 판정",
        "자원 계획",
        "메모리 설정",
        "최근 완료",
        "  ✓",
        "run별 최대",
        "단계 잔여 예상",
        "─",
    )
    while len(lines) > rows:
        index = next(
            (i for prefix in optional for i, line in enumerate(lines) if line.startswith(prefix)),
            None,
        )
        if index is None:
            break
        lines.pop(index)
    # Most terminals >=24 rows fit all16 workers. Tiny windows retain an explicit resize notice.
    if len(lines) > rows:
        lines = lines[: max(0, rows - 2)] + [
            "화면이 작습니다. 터미널 높이를 늘리거나 status 명령을 사용하세요.",
            lines[-1],
        ]
    return "\n".join(cell(line, max(1, columns - 1)).rstrip() for line in lines)
