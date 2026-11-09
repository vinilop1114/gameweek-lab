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
# Debajo de esta hora local el deadline cae de madrugada, y el aviso se
# manda la noche anterior: el modelo va a decidir mientras uno duerme.
EARLY_LOCAL_HOUR = 9
NIGHT_BEFORE_HOUR = 21
# Para los deadlines en horario razonable, cuanto antes avisar. 90 minutos
# deja al Action correr dentro de su ventana y todavia da tiempo a copiar.
LEAD_MINUTES = 90

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

        if deadline_local.hour < EARLY_LOCAL_HOUR:
            aviso_local = (deadline_local - timedelta(days=1)).replace(
                hour=NIGHT_BEFORE_HOUR, minute=0, second=0, microsecond=0
            )
            porque = (
                f"El deadline es a las {deadline_local:%H:%M} de tu manana, asi que el "
                f"modelo decide entre las {ventana_inicio:%H:%M} y las "
                f"{deadline_local:%H:%M} mientras dormis. Si la ganancia de algun chip "
                "esta cerca de su umbral, aplicalo a mano antes de ir a dormir."
            )
        else:
            aviso_local = deadline_local - timedelta(minutes=LEAD_MINUTES)
            porque = (
                f"El modelo ya decidio: su ventana es de "
                f"{ventana_inicio:%H:%M} a {deadline_local:%H:%M}."
            )

        aviso_utc = aviso_local.astimezone(timezone.utc)
        descripcion = _escape(
            f"Deadline GW{gameweek}: {deadline_local:%a %d %b %H:%M} (hora local).\n\n"
            "Que hacer: abrir data/processed/briefing.md y replicar en la app lo que "
            "diga, transferencias y chips. El modelo decide; aca solo se copia.\n\n"
            f"{porque}\n\n"
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
            f"DTSTART:{aviso_utc:%Y%m%dT%H%M%SZ}",
            f"DTEND:{(aviso_utc + timedelta(minutes=30)):%Y%m%dT%H%M%SZ}",
            f"SUMMARY:FPL GW{gameweek} - replicar lo que decidio el modelo",
            f"DESCRIPTION:{descripcion}",
            "BEGIN:VALARM",
            "TRIGGER:-PT15M",
            "ACTION:DISPLAY",
            f"DESCRIPTION:FPL GW{gameweek} - deadline {deadline_local:%H:%M}",
            "END:VALARM",
            "END:VEVENT",
        ]

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
    CALENDAR_PATH.write_text(ics, encoding="utf-8", newline="")
    print(
        f"Guardado {CALENDAR_PATH} "
        f"({ics.count('BEGIN:VEVENT')} recordatorios, zona UTC{args.utc_offset:+g})"
    )
