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
from gameweek_lab.squad_builder import (
    _best_xi_total,
    select_squad,
    select_starting_xi,
    select_wildcard_squad,
)
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
# Umbrales de los chips de UNA fecha. Los tres son juicios, igual que el
# del Wildcard: el costo real de gastar un chip es la mejor fecha en que
# podrías gastarlo después, y ponerle número a eso exige proyectar la
# temporada entera. Se eligieron mirando la distribución real de cada
# ganancia — ver `evaluate_one_week_chips` — y la rampa de vencimiento
# los baja igual que al Wildcard, porque un chip sin usar se pierde.
#
# Triple Captain: la ganancia es una copia extra del capitán, así que el
# umbral es directamente "cuánto tiene que proyectar el capitán".
TRIPLE_CAPTAIN_MIN_GAIN = 9.0
# Bench Boost: la ganancia son los 4 del banco. Con banco normal suma
# ~6-8, así que el umbral apunta a una fecha con banco cargado o doble.
BENCH_BOOST_MIN_GAIN = 14.0
# Free Hit: rehace el equipo por UNA fecha y después vuelve todo atrás.
# Compite con el Wildcard, que es permanente, así que exige más.
FREE_HIT_MIN_GAIN = 18.0
# Cuántas fechas antes del vencimiento empieza a bajar el umbral.
WILDCARD_RAMP_GAMEWEEKS = 4
# **Hay DOS sets de chips, uno por mitad de temporada.** El primero vence
# en el deadline de GW19 (2 enero 2027) y el segundo al cerrar la
# temporada. Un chip sin usar no se acumula: se pierde.
FIRST_HALF_LAST_GAMEWEEK = 19
SEASON_LAST_GAMEWEEK = 38

CHIP_STATE_KEY = "chips_used"


def season_half(gameweek: int) -> int:
    """1 o 2 según la mitad de temporada a la que pertenece la fecha."""
    return 1 if gameweek <= FIRST_HALF_LAST_GAMEWEEK else 2


def chip_expiry_gameweek(gameweek: int) -> int:
    """Última fecha en que se puede usar el chip del set vigente."""
    return FIRST_HALF_LAST_GAMEWEEK if season_half(gameweek) == 1 else SEASON_LAST_GAMEWEEK


def chip_token(chip: str, gameweek: int) -> str:
    """Cómo se guarda en el estado: `wildcard:1` / `wildcard:2`."""
    return f"{chip}:{season_half(gameweek)}"


def chip_available(chip: str, gameweek: int) -> bool:
    """Si queda el chip **del set de esta mitad**.

    El estado guarda tokens tipo `wildcard:1` / `wildcard:2`, porque son
    dos chips distintos: haber jugado el de la primera mitad no consume
    el de la segunda. La primera versión guardaba solo `"wildcard"` en
    una lista plana, así que usarlo en diciembre bloqueaba para siempre
    el de enero a junio — el chip de la segunda mitad quedaba muerto sin
    que nada lo avisara.

    **Lo marca el modelo al jugarlo** (ver `wildcard_decision`), no una
    persona. Si alguna vez hay que corregirlo a mano, es
    `chips_used` en `data/processed/base_squad_state.json`.

    Un token sin sufijo se interpreta como del primer set, por si quedó
    alguno escrito por la versión anterior.
    """
    used = _load_base_state().get(CHIP_STATE_KEY, [])
    legacy = chip in used and season_half(gameweek) == 1
    return not legacy and chip_token(chip, gameweek) not in used


def wildcard_threshold(gameweek: int) -> float:
    """Ganancia exigida en `gameweek`, con la rampa de vencimiento del
    set vigente.

    La rampa se mide contra el vencimiento de **esta** mitad, no contra
    un número fijo. Antes el vencimiento estaba cableado en GW19, así que
    desde GW20 el umbral quedaba en 0 **para el resto de la temporada**:
    el chip de la segunda mitad se habría jugado en la primera fecha con
    cualquier ganancia positiva, cuando todavía faltaban 18 fechas para
    que venciera. Justo lo contrario de lo que la rampa busca.
    """
    expiry = chip_expiry_gameweek(gameweek)
    if gameweek >= expiry:
        return 0.0
    ramp_start = expiry - WILDCARD_RAMP_GAMEWEEKS
    if gameweek < ramp_start:
        return WILDCARD_MIN_GAIN
    return round(WILDCARD_MIN_GAIN * (expiry - gameweek) / WILDCARD_RAMP_GAMEWEEKS, 2)


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
        "gameweeks_to_expiry": chip_expiry_gameweek(current_gw) - current_gw,
        "season_half": season_half(current_gw),
        "expiry_gameweek": chip_expiry_gameweek(current_gw),
    }


def wildcard_decision(players: pd.DataFrame, team_path: str = "my_team.csv"):
    """Si corresponde jugar el Wildcard ahora, devuelve `(plantel_nuevo,
    log, token)`; si no, `None`. El token es lo que se guarda en
    `chips_used` — lleva la mitad de temporada, porque son dos chips
    distintos.

    Lo llama `evolve_base_squad` **después** de sus dos frenos (ventana
    del deadline y una-vez-por-fecha) y **antes** de evaluar
    transferencias, cortando ahí: un Wildcard rehace los 15, así que
    evaluar transferencias encima no tendría sentido.

    Ejecuta, igual que las transferencias, porque el flujo real es que
    una persona copia en la app exactamente lo que el modelo decidió.
    Dejarlo en "recomendación" obligaba a marcar el chip a mano y abría
    el modo de falla asimétrico: si no lo marcabas, el briefing te lo
    recomendaba para siempre.

    Al jugarlo, las transferencias acumuladas se pierden: FPL da cambios
    ilimitados esa fecha y arrancás de cero, con 1 libre en la siguiente.
    """
    current_gw = _current_gameweek(players)
    if not chip_available("wildcard", current_gw):
        return None

    verdict = evaluate_wildcard(players, team_path)
    if not verdict["recommend"]:
        return None

    squad = verdict["wildcard_squad"].copy()
    # Comprados hoy: su precio de venta futuro se calcula desde acá.
    squad["purchase_price"] = squad["now_cost"]

    previous = set(load_my_team(team_path, players)["web_name"])
    incoming = sorted(set(squad["web_name"]) - previous)
    outgoing = sorted(previous - set(squad["web_name"]))
    log = [
        f"WILDCARD (set {verdict['season_half']}) JUGADO en GW{verdict['gameweek']}: +{verdict['gain']:.2f} xP en "
        f"{HORIZON_GAMEWEEKS} fechas sobre seguir con transferencias normales "
        f"(umbral {verdict['threshold']}).",
        f"Salen ({len(outgoing)}): {', '.join(outgoing)}",
        f"Entran ({len(incoming)}): {', '.join(incoming)}",
    ]
    return squad, log, chip_token("wildcard", current_gw)


ONE_WEEK_CHIPS = {
    "triple_captain": TRIPLE_CAPTAIN_MIN_GAIN,
    "bench_boost": BENCH_BOOST_MIN_GAIN,
    "free_hit": FREE_HIT_MIN_GAIN,
}


def _squad_budget(squad: pd.DataFrame) -> float:
    return round(sum(_sell_price(r.now_cost, r.purchase_price) for r in squad.itertuples()), 1)


def evaluate_one_week_chips(players: pd.DataFrame, team_path: str = "my_team.csv") -> list[dict]:
    """Los tres chips que valen por UNA fecha: Triple Captain, Bench
    Boost y Free Hit.

    A diferencia del Wildcard, ninguno cambia el plantel de forma
    permanente, así que **ninguno toca `my_team.csv`**. Lo que hacen es
    marcarse como usados y publicar qué aplicar en la app:

    - **Triple Captain**: el capitán puntúa ×3 en vez de ×2, así que la
      ganancia es una copia extra de él. Se mide sobre el XI ya elegido,
      no sobre el plantel: la cinta solo puede ir a un titular.
    - **Bench Boost**: puntúan los 4 del banco. La ganancia es la suma de
      sus xP — nada más, porque el XI no cambia.
    - **Free Hit**: rehace el equipo por una fecha y **vuelve atrás** a la
      siguiente. Como el banco no se sostiene, el plantel correcto es el
      de `select_wildcard_squad` (banco mínimo, XI máximo) y no el de
      `select_squad`: es exactamente el ejercicio teórico que este repo
      venía calculando desde el principio sin tener dónde usarlo.

    La ganancia de los tres se mide con `xp_next`, que es la señal
    medida para una fecha (Spearman 0.633 sobre GW6-GW10), y no con el
    horizonte: estos chips no tienen efecto más allá de su fecha.
    """
    squad = load_my_team(team_path, players)
    starters, bench = select_starting_xi(squad)
    current_gw = _current_gameweek(players)
    threshold_scale = wildcard_threshold(current_gw) / WILDCARD_MIN_GAIN

    captain = starters.nlargest(1, "xp_next").iloc[0]
    free_hit_starters, _ = select_wildcard_squad(players, budget=_squad_budget(squad))

    gains = {
        "triple_captain": (
            round(float(captain.xp_next), 2),
            f"Capitán triple: {captain.web_name} ({captain.team_name}) vs {captain.next_opponent}.",
        ),
        "bench_boost": (
            round(float(bench["xp_next"].sum()), 2),
            "Banco: " + ", ".join(f"{b.web_name} {b.xp_next}" for b in bench.itertuples()),
        ),
        "free_hit": (
            round(float(free_hit_starters["xp_next"].sum() - starters["xp_next"].sum()), 2),
            "XI de una fecha: "
            + ", ".join(free_hit_starters.nlargest(5, "xp_next")["web_name"])
            + ", …",
        ),
    }

    results = []
    for chip, base_threshold in ONE_WEEK_CHIPS.items():
        if not chip_available(chip, current_gw):
            continue
        gain, detail = gains[chip]
        # La rampa de vencimiento se aplica en proporción, igual que al
        # Wildcard: cada chip tiene su propio umbral base, pero todos
        # pierden exigencia a la misma velocidad cerca del cierre del set.
        threshold = round(base_threshold * threshold_scale, 2)
        results.append({
            "chip": chip,
            "gain": gain,
            "threshold": threshold,
            "recommend": gain >= threshold,
            "detail": detail,
            "squad": free_hit_starters if chip == "free_hit" else None,
        })
    return results


def one_week_chip_decision(players: pd.DataFrame, team_path: str = "my_team.csv"):
    """El mejor chip de una fecha que supere su umbral, o `None`.

    Devuelve `(log, token)`. **No devuelve plantel**: ninguno de los tres
    cambia `my_team.csv`. Si más de uno califica se elige el de mayor
    ganancia relativa a su umbral — jugar dos en la misma fecha es legal
    en FPL pero gastar dos de golpe rara vez es lo mejor, y el que quede
    sigue disponible la semana siguiente.
    """
    current_gw = _current_gameweek(players)
    candidates = [c for c in evaluate_one_week_chips(players, team_path) if c["recommend"]]
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["gain"] / max(c["threshold"], 0.01))
    nombre = best["chip"].replace("_", " ").upper()
    log = [
        f"{nombre} (set {season_half(current_gw)}) JUGADO en GW{current_gw}: "
        f"+{best['gain']:.2f} xP (umbral {best['threshold']}).",
        best["detail"],
        "No cambia el plantel — se activa el chip en la app y nada más.",
    ]
    return log, chip_token(best["chip"], current_gw)
