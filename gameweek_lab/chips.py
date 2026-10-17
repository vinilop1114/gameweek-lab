"""¿Conviene jugar un chip? — evaluación sobre el horizonte, no sobre la fecha.

Hasta acá el sistema no podía recomendar un chip: no guardaba cuáles
quedaban, `evolve_base_squad` solo sabía hacer transferencias, y el
`squad_type = "Wildcard"` del CSV era un techo teórico que aparecía igual
tuviera o no sentido jugarlo (rearma desde cero cada corrida y vacía el
banco a propósito, así que no es sostenible).

Lo que se evalúa acá es otra cosa: **un Wildcard jugable**, optimizado
sobre las próximas `HORIZON_GAMEWEEKS` fechas y con tu presupuesto real,
contra **el Base tal como evolucionaría** haciendo transferencias
normales en esas mismas fechas. Esa es la comparación honesta: el Base no
se queda quieto, así que medir contra el Base congelado inflaría la
ganancia del chip en todo lo que las transferencias normales habrían
conseguido gratis.

Limitación heredada y grande: el horizonte sale de `xp_horizon`, que lo
calcula el motor propio — el mismo que la calibración mide como el peor
predictor de la tabla (0.359 contra 0.627 del modelo vigente). Mientras
el horizonte no tenga su propia medición (empezó a grabarse en GW6, la
primera lectura es de mediados de noviembre), **esta recomendación
descansa sobre la parte más débil del sistema**. Por eso recomienda y no
ejecuta.
"""

import pandas as pd

from gameweek_lab.analysis import HORIZON_GAMEWEEKS, gameweek_expected_points
from gameweek_lab.squad_builder import _best_xi_total, select_squad
from gameweek_lab.transfer_advisor import (
    MAX_BANKED_TRANSFERS,
    _current_gameweek,
    _load_base_state,
    _sell_price,
    _transfers_granted,
    load_my_team,
    plan_transfers,
)

# Cuánto xP tiene que ganar el Wildcard sobre el horizonte para que valga
# quemarlo. Es un JUICIO, no una derivación: el costo real de usarlo es la
# mejor fecha en que podrías usarlo más adelante, y eso exige proyectar la
# temporada entera. 15 xP en 4 fechas (~3.75 por fecha) es
# deliberadamente alto — la comparación ya descuenta lo que las
# transferencias normales consiguen gratis, así que lo que queda es
# incremental, y un chip es de un solo uso.
WILDCARD_MIN_GAIN = 15.0
# Desde esta fecha el umbral baja en rampa hasta cero en el vencimiento.
# Un chip vencido vale 0, así que cerca del final conviene bajar el
# estándar de "momento perfecto" — la regla ya estaba escrita a mano en
# docs/chips-strategy.md y acá se vuelve ejecutable.
WILDCARD_RAMP_START_GAMEWEEK = 15
# El primer set de chips expira en el deadline de GW19 (2 enero 2027).
WILDCARD_EXPIRY_GAMEWEEK = 19

CHIP_STATE_KEY = "chips_used"


def chip_available(chip: str = "wildcard") -> bool:
    """Si el chip sigue disponible, según `chips_used` en el estado.

    **Lo marca una persona, no el sistema.** Jugar un chip es una acción
    manual en la app de FPL y la API pública no expone qué chips usó un
    equipo, así que no hay forma de detectarlo. Si lo jugaste, agregalo a
    `chips_used` en `data/processed/base_squad_state.json`; si no, el
    briefing va a seguir recomendándotelo para siempre.
    """
    return chip not in _load_base_state().get(CHIP_STATE_KEY, [])


def wildcard_threshold(gameweek: int) -> float:
    """Umbral de ganancia exigido en `gameweek`, con la rampa de
    vencimiento aplicada."""
    if gameweek >= WILDCARD_EXPIRY_GAMEWEEK:
        return 0.0
    if gameweek < WILDCARD_RAMP_START_GAMEWEEK:
        return WILDCARD_MIN_GAIN
    span = WILDCARD_EXPIRY_GAMEWEEK - WILDCARD_RAMP_START_GAMEWEEK
    remaining = WILDCARD_EXPIRY_GAMEWEEK - gameweek
    return round(WILDCARD_MIN_GAIN * remaining / span, 2)


def _week_xi_points(squad: pd.DataFrame, weekly_by_id: dict, week: int) -> float:
    """Puntos del mejor XI de ese plantel en esa fecha. Reusa la misma
    aritmética que el asesor de transferencias (`_best_xi_total`), así
    que un plantel se puntúa igual acá que allá."""
    values = {
        position: sorted(
            (weekly_by_id[pid][week] for pid in squad.loc[squad["position"] == position, "id"]),
            reverse=True,
        )
        for position in ("GKP", "DEF", "MID", "FWD")
    }
    return _best_xi_total(values)[0]


def _fixed_squad_points(squad: pd.DataFrame, weekly_by_id: dict, weeks: int) -> float:
    return sum(_week_xi_points(squad, weekly_by_id, w) for w in range(weeks))


def _evolving_base_points(
    players: pd.DataFrame, squad: pd.DataFrame, weekly_by_id: dict, weeks: int
) -> tuple[float, list[str]]:
    """Lo que rinde el Base si NO se juega el chip: semana a semana,
    haciendo las transferencias que el modelo haría igual.

    Los hits se descuentan del total (4 puntos cada uno). Si el Base
    llega a donde llega pagando hits, eso es parte de su costo y el
    Wildcard se los ahorra — no contarlos regalaría ventaja al Base.
    """
    state = _load_base_state()
    current_gw = _current_gameweek(players)
    free_transfers = state["banked_free_transfers"]
    total, log = 0.0, []

    for offset in range(weeks):
        gameweek = current_gw + offset
        ya_ejecutada = offset == 0 and state["last_evaluated_gameweek"] == current_gw
        if not ya_ejecutada:
            free_transfers = min(
                free_transfers + _transfers_granted(gameweek), MAX_BANKED_TRANSFERS
            )
            squad, free_transfers, week_log = plan_transfers(
                players, squad, free_transfers, gameweek
            )
            log += week_log
            total -= 4 * sum("HIT -4" in line for line in week_log)
        total += _week_xi_points(squad, weekly_by_id, offset)
    return total, log


def evaluate_wildcard(
    players: pd.DataFrame, team_path: str = "my_team.csv", weeks: int = HORIZON_GAMEWEEKS
) -> dict:
    """Compara jugar el Wildcard ahora contra seguir con transferencias
    normales, sobre las próximas `weeks` fechas.

    Devuelve un dict con la ganancia, el umbral vigente, el veredicto y
    los planteles, para que el briefing lo publique. **No ejecuta nada**:
    jugar un chip es una acción manual en la app de FPL, y rehacer 15
    jugadores de golpe sobre `my_team.csv` es demasiado irreversible para
    que lo dispare un cron sin que nadie lo mire.
    """
    squad = load_my_team(team_path, players)
    current_gw = _current_gameweek(players)
    weekly = gameweek_expected_points(players, weeks)
    weekly_by_id = dict(zip(players["id"], weekly.to_numpy().tolist()))

    # Presupuesto real: lo que recuperarías vendiendo los 15. Usar los
    # £100m del arranque inventaría plata si tu equipo perdió valor, o la
    # escondería si ganó.
    budget = round(
        sum(_sell_price(r.now_cost, r.purchase_price) for r in squad.itertuples()), 1
    )

    base_points, base_log = _evolving_base_points(players, squad.copy(), weekly_by_id, weeks)
    wildcard_squad = select_squad(players, xp_column="xp_horizon", budget=budget)
    # El Wildcard se puntúa como plantel FIJO: en la vida real también
    # podría hacer transferencias después, así que esto lo subestima a
    # propósito. Es el lado conservador para una decisión de un solo uso.
    wildcard_points = _fixed_squad_points(wildcard_squad, weekly_by_id, weeks)

    gain = round(wildcard_points - base_points, 2)
    threshold = wildcard_threshold(current_gw)
    return {
        "gameweek": current_gw,
        "budget": budget,
        "base_points": round(base_points, 2),
        "wildcard_points": round(wildcard_points, 2),
        "gain": gain,
        "threshold": threshold,
        "recommend": gain >= threshold,
        "base_log": base_log,
        "wildcard_squad": wildcard_squad,
        "gameweeks_to_expiry": WILDCARD_EXPIRY_GAMEWEEK - current_gw,
    }
