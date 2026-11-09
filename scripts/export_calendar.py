"""Genera un .ics con los recordatorios para replicar lo que decide el modelo.

Se genera desde `bootstrap-static.json`, no a mano: los deadlines de FPL
se mueven (partidos postergados, cambios de TV) y un calendario escrito a
mano queda desfasado en silencio. El anterior tenia la ultima fecha mal
configurada, y `docs/chips-strategy.md` todavia decia que el primer set
de chips vencia el sabado 2 de enero a las 13:30 GMT cuando la API dice
viernes 1 a las 18:30 UTC: un dia antes y cinco horas mas tarde.

La hora del recordatorio NO es la del deadline. El modelo decide en las
3 horas previas, y en zona UTC-6 varios deadlines caen entre las 02:00 y
las 07:30 locales: el aviso tiene que llegar cuando todavia se puede
hacer algo, no cuando ya paso.

Uso:
    python scripts/export_calendar.py
    python scripts/export_calendar.py --utc-offset -5
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.chips import (  # noqa: E402
    FIRST_HALF_LAST_GAMEWEEK,
    SEASON_LAST_GAMEWEEK,
    WILDCARD_RAMP_GAMEWEEKS,
    chip_available,
    season_half,
)
from gameweek_lab.config import DATA_PROCESSED_DIR, DATA_RAW_DIR  # noqa: E402
from gameweek_lab.transfer_advisor import TRANSFER_DECISION_WINDOW_HOURS  # noqa: E402

CALENDAR_PATH = DATA_PROCESSED_DIR / "fpl_deadlines.ics"
# El evento va en la hora EXACTA del deadline, y los avisos son alarmas
# del evento. Una version anterior corria el evento a la hora en que
# convenia avisar; eso ponia el recordatorio en el calendario pero dejaba
# el deadline real invisible, que es el dato que uno quiere ver.
EVENT_MINUTES = 30
# Dos alarmas por evento: una con un dia de anticipacion para ver venir la
# fecha, y otra a 3 horas, que es cuando se abre la ventana de decision
# del modelo (TRANSFER_DECISION_WINDOW_HOURS).
ALARM_TRIGGERS = ("-P1D", f"-PT{TRANSFER_DECISION_WINDOW_HOURS}H")

CHIP_NAMES = {
    "wildcard": "Wildcard",
    "triple_captain": "Triple Captain",
    "bench_boost": "Bench Boost",
    "free_hit": "Free Hit",
}


def _escape(text: str) -> str:
    """RFC 5545: hay que escapar la barra invertida, la coma, el punto y
    coma y los saltos de linea. El orden importa: la barra primero, o se
    re-escaparian las barras que agregan los demas reemplazos."""
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """Plegado a 75 octetos (RFC 5545).

    Se cuenta en BYTES y no en caracteres: con acentos una linea de 75
    caracteres puede pasarse de 75 octetos. Y el corte no puede caer en
    medio de un caracter multibyte, o el cliente lo lee como basura.
    """
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    chunks, rest = [], raw
    while True:
        # La primera linea puede usar los 75 octetos; las de continuacion
        # llevan un espacio inicial que TAMBIEN cuenta para el limite, asi
        # que solo les quedan 74 de contenido. Sin esto el archivo salia
        # con 156 lineas de 76 octetos: la mayoria de los clientes lo
        # tolera, pero esta fuera de la RFC.
        budget = 75 if not chunks else 74
        if len(rest) <= budget:
            chunks.append(rest.decode("utf-8"))
            break
        cut = budget
        while cut > 0 and (rest[cut] & 0xC0) == 0x80:
            cut -= 1
        chunks.append(rest[:cut].decode("utf-8"))
        rest = rest[cut:]
    return "\r\n ".join(chunks)


def _chips_sin_usar(half: int) -> str:
    probe = 1 if half == 1 else SEASON_LAST_GAMEWEEK
    libres = [name for chip, name in CHIP_NAMES.items() if chip_available(chip, probe)]
    return ", ".join(libres) if libres else "ninguno"


def _nota_umbral(gameweek: int) -> str:
    half = season_half(gameweek)
    expiry = FIRST_HALF_LAST_GAMEWEEK if half == 1 else SEASON_LAST_GAMEWEEK
    if gameweek >= expiry:
        return (
            f"UMBRAL EN CERO: ultima fecha del set {half}. El modelo juega cualquier "
            "chip que quede sin usar. Lo que no se juegue hoy se pierde."
        )
    restantes = expiry - gameweek
    if restantes > WILDCARD_RAMP_GAMEWEEKS:
        return "Umbral de chips completo: solo se juegan si la ganancia es clara."
    escala = restantes / WILDCARD_RAMP_GAMEWEEKS
    return (
        f"Umbral de chips al {escala:.0%} (quedan {restantes} fechas del set {half}). "
        "Cuanto mas cerca del cierre, menos exigente."
    )


def _sin_sellos(ics: str) -> str:
    """El .ics sin las líneas que cambian solas en cada generación, para
    poder comparar si cambió algo de fondo."""
    return "\r\n".join(
        line
        for line in ics.split("\r\n")
        if not line.startswith(("DTSTAMP:", "SEQUENCE:", "X-WR-CALDESC:"))
    )


def build_calendar(utc_offset_hours: float) -> str:
    local = timezone(timedelta(hours=utc_offset_hours))
    bootstrap = json.loads(
        (DATA_RAW_DIR / "bootstrap-static.json").read_text(encoding="utf-8")
    )
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    # Minutos desde el arranque de la temporada: monótono y estable entre
    # corridas, así que una generación posterior siempre gana sobre la
    # anterior en el cliente de calendario.
    sequence = int((now - datetime(2026, 8, 1, tzinfo=timezone.utc)).total_seconds() // 60)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//The Gameweek Lab//Deadlines//ES",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:FPL - The Gameweek Lab",
        "X-WR-CALDESC:" + _escape(
            f"Generado {stamp} desde la API de FPL. Regenerar con "
            "scripts/export_calendar.py cuando FPL mueva un deadline."
        ),
    ]

    for event in bootstrap["events"]:
        deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
        if deadline <= now:
            continue

        gameweek = event["id"]
        deadline_local = deadline.astimezone(local)
        ventana_inicio = deadline_local - timedelta(hours=TRANSFER_DECISION_WINDOW_HOURS)
        madrugada = deadline_local.hour < 9

        nota_horario = (
            f"Ojo: el modelo decide entre las {ventana_inicio:%H:%M} y las "
            f"{deadline_local:%H:%M} de tu manana, asi que la alarma de 3 horas va a "
            "sonar de madrugada. La de 24 horas es la util: si la ganancia de algun "
            "chip esta cerca de su umbral, conviene aplicarlo a mano con tiempo."
            if madrugada
            else f"El modelo decide entre las {ventana_inicio:%H:%M} y las "
            f"{deadline_local:%H:%M}, asi que a 3 horas puede no haber decidido "
            "todavia: el Action corre cada 3 horas."
        )

        descripcion = _escape(
            f"Deadline GW{gameweek}: {deadline_local:%a %d %b %H:%M} (hora local).\n"
            "Avisos: 24 horas antes y 3 horas antes.\n\n"
            "Que hacer: abrir data/processed/briefing.md y replicar en la app lo que "
            "diga, transferencias y chips. El modelo decide; aca solo se copia.\n\n"
            f"{nota_horario}\n\n"
            f"{_nota_umbral(gameweek)}\n"
            f"Chips sin usar del set {season_half(gameweek)}: "
            f"{_chips_sin_usar(season_half(gameweek))}."
        )

        lines += [
            "BEGIN:VEVENT",
            # El UID es ESTABLE por fecha, sin la data del deadline: si FPL
            # mueve GW11, al re-importar el cliente actualiza ese evento en
            # vez de crear un duplicado. Con la fecha adentro, cada cambio
            # de calendario dejaba dos avisos para la misma fecha.
            f"UID:gw{gameweek}@gameweek-lab",
            f"DTSTAMP:{stamp}",
            # SEQUENCE sube en cada generación para que el cliente acepte
            # la versión nueva como posterior a la que ya tenía.
            f"SEQUENCE:{sequence}",
            f"DTSTART:{deadline:%Y%m%dT%H%M%SZ}",
            f"DTEND:{(deadline + timedelta(minutes=EVENT_MINUTES)):%Y%m%dT%H%M%SZ}",
            f"SUMMARY:Deadline FPL GW{gameweek}",
            f"DESCRIPTION:{descripcion}",
        ]
        for trigger in ALARM_TRIGGERS:
            cuanto = "24 horas" if trigger == "-P1D" else f"{TRANSFER_DECISION_WINDOW_HOURS} horas"
            lines += [
                "BEGIN:VALARM",
                f"TRIGGER:{trigger}",
                "ACTION:DISPLAY",
                f"DESCRIPTION:" + _escape(
                    f"FPL GW{gameweek}: faltan {cuanto} para el deadline "
                    f"({deadline_local:%a %d %b %H:%M}). Revisar briefing.md."
                ),
                "END:VALARM",
            ]
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utc-offset",
        type=float,
        default=-6.0,
        help="Desplazamiento horario para calcular las horas de aviso (default -6).",
    )
    args = parser.parse_args()

    CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ics = build_calendar(args.utc_offset)
    eventos = ics.count("BEGIN:VEVENT")

    # Solo se reescribe si cambió algo REAL. `DTSTAMP` y `SEQUENCE` llevan
    # la hora de generación, así que sin esta comparación el archivo salía
    # distinto en cada corrida y el Action commiteaba ocho veces por día un
    # calendario idéntico. Comparar ignorándolos deja además la semántica
    # correcta: `SEQUENCE` sube solo cuando el contenido cambió de verdad,
    # que es justo lo que un cliente de calendario espera.
    if CALENDAR_PATH.exists() and _sin_sellos(
        CALENDAR_PATH.read_bytes().decode("utf-8")
    ) == _sin_sellos(ics):
        print(f"{CALENDAR_PATH} sin cambios ({eventos} recordatorios) — no se reescribe.")
        raise SystemExit(0)

    CALENDAR_PATH.write_text(ics, encoding="utf-8", newline="")
    print(
        f"Guardado {CALENDAR_PATH} "
        f"({eventos} recordatorios, zona UTC{args.utc_offset:+g})"
    )
