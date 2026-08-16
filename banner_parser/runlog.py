"""Диагностическое логирование: старт, сигналы, heartbeat, причина завершения.

Задача модуля — сделать так, чтобы причина остановки процесса читалась из лога
без гадания. Как читать посмертный лог:

  • есть строка «=== ЗАВЕРШЕНО» — процесс отработал сам, в строке код возврата;
  • есть строка «=== СИГНАЛ»    — процесс попросили остановиться извне
                                  (kill, Ctrl-C, SIGHUP при обрыве SSH);
  • есть строка «=== ИСКЛЮЧЕНИЕ» — упал с трассировкой, она тут же ниже;
  • лог обрывается без единой из них — процесс убит SIGKILL, перехватить его
    нельзя. Два реальных источника: OOM-killer и shared-хостинг, который сносит
    всю сессию при выходе из SSH (nohup/setsid/screen/tmux не спасают).
    Последняя строка «жив:» показывает этап и память на момент смерти —
    по ней видно, память это была или нет.

Всё, что пишет этот модуль, идёт в лог немедленно: logging.FileHandler делает
flush после каждой записи, поэтому хвост лога не теряется даже при SIGKILL.
Чего нельзя сказать про print() в перенаправленный файл — поэтому запускать
процесс нужно только через `python3 -u` (см. scripts/run_crawl.sh).
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import platform
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("banner_parser.run")

_START = time.monotonic()
_STAGE = "старт"
_STAGE_SINCE = time.monotonic()
_STAGE_LOCK = threading.Lock()
_HEARTBEAT: Optional[threading.Thread] = None
_HEARTBEAT_STOP = threading.Event()
_FINISHED = False          # выставляется, когда причина завершения уже записана


# ---- метрики процесса ----------------------------------------------------
def rss_mb() -> float:
    """Текущий RSS процесса в МБ (0.0, если платформа не даёт узнать)."""
    try:                                    # Linux — самый точный путь
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:                                    # POSIX-фолбэк: пик, не текущий
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                       # noqa: BLE001 — Windows и прочее
        return 0.0


def peak_rss_mb() -> float:
    """Пиковый RSS процесса в МБ (0.0, если недоступен)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                       # noqa: BLE001
        return 0.0


def rss_str() -> str:
    """RSS для лога. На Windows метрика недоступна — печатать 0 МБ было бы
    враньём, поэтому честное «n/d»."""
    v = rss_mb()
    return f"{v:.0f} МБ" if v > 0 else "n/d"


def peak_rss_str() -> str:
    v = peak_rss_mb()
    return f"{v:.0f} МБ" if v > 0 else "n/d"


def _mem_available_mb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _disk_free_mb(path: str = ".") -> Optional[float]:
    try:
        return shutil.disk_usage(path).free / (1024.0 * 1024.0)
    except OSError:
        return None


def _pkg_version(name: str) -> str:
    """Версия пакета без его импорта (импорт torch — это секунды и гигабайты)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return "нет"
    except Exception:                       # noqa: BLE001
        return "?"


# ---- настройка логирования -----------------------------------------------
_NOISY = ("httpx", "urllib3", "transformers", "PIL", "filelock", "huggingface_hub",
          "matplotlib", "asyncio")


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> Optional[Path]:
    """Логирование в stderr и (опционально) в файл. Возвращает путь к файлу.

    Файловый обработчик flush'ит каждую запись — хвост лога переживает SIGKILL.
    """
    # Консоль Windows по умолчанию в cp866/cp1251, и любой не-ASCII символ в
    # print() роняет команду с UnicodeEncodeError — так падал export на стрелке
    # «→», хотя файл уже был записан. Просим потоки не падать на таких символах.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):           # повторный вызов не должен дублировать
        root.removeHandler(h)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    path: Optional[Path] = None
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return path


def log_startup(argv: Optional[list[str]] = None, cfg_path: Optional[str] = None,
                log_path: Optional[Path] = None) -> None:
    """Шапка запуска: всё, чего не хватало, когда падало «No module named ...».

    Печатается ДО тяжёлых импортов, поэтому доезжает до лога даже если процесс
    умрёт на загрузке модели.
    """
    cwd = os.getcwd()
    log.info("=== СТАРТ banner_parser ===")
    log.info("команда   : %s", " ".join(argv or sys.argv))
    log.info("каталог   : %s", cwd)
    log.info("python    : %s (%s)", sys.executable, platform.python_version())
    log.info("пакет     : %s", Path(__file__).resolve().parent)
    log.info("конфиг    : %s", cfg_path or "config.yaml (по умолчанию)")
    log.info("pid/ppid  : %s / %s   хост: %s", os.getpid(), os.getppid(),
             platform.node())
    if log_path:
        log.info("лог-файл  : %s", log_path.resolve())

    env = {k: os.environ.get(k) for k in
           ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "HF_HOME", "OMP_NUM_THREADS")}
    log.info("окружение : %s", ", ".join(f"{k}={v or '—'}" for k, v in env.items()))

    versions = ", ".join(f"{p}={_pkg_version(p)}" for p in
                         ("torch", "ultralytics", "easyocr", "transformers", "osmnx"))
    log.info("пакеты    : %s", versions)

    free = _disk_free_mb(cwd)
    avail = _mem_available_mb()
    log.info("ресурсы   : свободно на диске %s, доступно памяти %s",
             f"{free/1024:.1f} ГБ" if free is not None else "?",
             f"{avail/1024:.1f} ГБ" if avail is not None else "?")

    # Предупреждаем заранее: это ровно те грабли, на которых уже падали.
    # Предупреждаем только если ultralytics реально стоит — иначе это ложная
    # тревога: без пакета переменная ни на что не влияет.
    if not os.environ.get("YOLO_CONFIG_DIR") and _pkg_version("ultralytics") != "нет":
        log.warning("YOLO_CONFIG_DIR не задан — ultralytics попробует писать в "
                    "~/.config/Ultralytics; если он недоступен на запись, будет "
                    "ошибка. Ставьте YOLO_CONFIG_DIR=/tmp/Ultralytics")
    if free is not None and free < 1024:
        log.warning("на диске меньше 1 ГБ (%.0f МБ) — загрузка весов моделей может "
                    "оборваться на середине", free)
    log.info("если лог оборвётся без строки ЗАВЕРШЕНО/СИГНАЛ/ИСКЛЮЧЕНИЕ — "
             "процесс убит SIGKILL (OOM или снос сессии хостингом)")
    log.info("=== начинаем работу ===")


# ---- этапы и heartbeat ----------------------------------------------------
def set_stage(stage: str) -> None:
    """Отметить текущий этап — попадёт в heartbeat и в строку завершения."""
    global _STAGE, _STAGE_SINCE
    with _STAGE_LOCK:
        _STAGE = stage
        _STAGE_SINCE = time.monotonic()


def current_stage() -> tuple[str, float]:
    with _STAGE_LOCK:
        return _STAGE, time.monotonic() - _STAGE_SINCE


def start_heartbeat(interval: float = 60.0) -> None:
    """Фоновая строка «жив: этап=..., rss=...» раз в interval секунд.

    Нужна ровно для одного: когда процесс убьют SIGKILL, последняя такая строка
    покажет, на каком этапе и с какой памятью он был в этот момент.
    """
    global _HEARTBEAT
    if _HEARTBEAT is not None or interval <= 0:
        return

    def _beat() -> None:
        while not _HEARTBEAT_STOP.wait(interval):
            stage, held = current_stage()
            log.info("жив: этап=%s (%.0f с), rss=%s, работает %.0f мин",
                     stage, held, rss_str(), (time.monotonic() - _START) / 60)

    _HEARTBEAT = threading.Thread(target=_beat, name="heartbeat", daemon=True)
    _HEARTBEAT.start()


def stop_heartbeat() -> None:
    _HEARTBEAT_STOP.set()


# ---- причина завершения ---------------------------------------------------
_SIGNAL_HINTS = {
    "SIGTERM": "kill без -9, timeout, или менеджер процессов хостинга",
    "SIGINT": "Ctrl-C",
    "SIGHUP": "оборвался терминал/SSH-сессия",
    "SIGQUIT": "Ctrl-\\ или явный kill -QUIT",
}


def _on_signal(signum: int, _frame) -> None:
    global _FINISHED
    name = signal.Signals(signum).name
    stage, held = current_stage()
    log.error("=== СИГНАЛ %s (%d): %s ===", name, signum,
              _SIGNAL_HINTS.get(name, "внешняя остановка"))
    log.error("оборван на этапе «%s» (%.0f с), rss=%s, всего работал %.0f мин",
              stage, held, rss_str(), (time.monotonic() - _START) / 60)
    _FINISHED = True
    stop_heartbeat()
    # Именно sys.exit, а не os._exit: SystemExit разматывает стек, поэтому
    # отработают finally-блоки — обход допишет итог и закроет БД.
    # Код выхода по конвенции 128+N — чтобы обёртка в shell увидела причину.
    sys.exit(128 + signum)


def _on_exception(exc_type, exc, tb) -> None:
    global _FINISHED
    if issubclass(exc_type, KeyboardInterrupt):
        log.error("=== СИГНАЛ SIGINT: Ctrl-C, остановлено вручную ===")
        _FINISHED = True
        stop_heartbeat()
        return
    stage, held = current_stage()
    log.critical("=== ИСКЛЮЧЕНИЕ на этапе «%s» (%.0f с), rss=%s ===",
                 stage, held, rss_str(), exc_info=(exc_type, exc, tb))
    _FINISHED = True


def _on_exit() -> None:
    if _FINISHED:
        return
    stop_heartbeat()
    log.info("=== ЗАВЕРШЕНО: работал %.1f мин, пик памяти %s, последний этап «%s» ===",
             (time.monotonic() - _START) / 60, peak_rss_str(), current_stage()[0])


def install_crash_handlers() -> None:
    """Ловим всё, что вообще ловится: сигналы, исключения, жёсткие падения."""
    # SIGSEGV/SIGABRT/SIGFPE/SIGBUS — трассировка прямо в stderr.
    faulthandler.enable()
    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:                     # Windows: SIGHUP/SIGQUIT отсутствуют
            continue
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):       # не главный поток / сигнал недоступен
            pass
    # SIGUSR1 — «покажи, где ты сейчас»: диагностика зависаний без отладчика.
    usr1 = getattr(signal, "SIGUSR1", None)
    if usr1 is not None:
        try:
            faulthandler.register(usr1, all_threads=True)
        except (OSError, ValueError):
            pass
    sys.excepthook = _on_exception
    atexit.register(_on_exit)
