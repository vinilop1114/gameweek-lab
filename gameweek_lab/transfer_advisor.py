import difflib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gameweek_lab.analysis import (
    HORIZON_GAMEWEEKS,
    MIN_MINUTES_FOR_RANKING,
    add_expected_points,
    add_horizon_expected_points,
    add_rank_adjusted_value,
    captaincy_picks,
    effective_minutes,
    gameweek_expected_points,
)
from gameweek_lab.build_dataset import build_players_dataset, get_next_deadline, get_team_fixtures_horizon
from gameweek_lab.config import DATA_PROCESSED_DIR
from gameweek_lab.squad_builder import (
    MAX_PER_CLUB,
    SQUAD_COMPOSITION,
    select_starting_xi,
    xi_gain_context,
    xi_horizon_gain,
)

# Estado persistente del equipo Base autogestionado (ver evolve_base_squad).
# Vive en data/processed porque el workflow de GitHub Actions ya commitea
# esa carpeta a diario — así el estado sobrevive entre corridas.
BASE_STATE_PATH = DATA_PROCESSED_DIR / "base_squad_state.json"

HIT_COST = 4.0
# Un hit se recomienda solo si la ganancia proyectada lo supera por este
# margen. El xP es una heurística con error real — pagar -4 puntos ciertos
# por una ventaja proyectada de 4.5 es apostar el margen de error entero.
HIT_UNCERTAINTY_MARGIN = 2.0
# Si el mejor cambio libre gana menos que esto en todo el horizonte, mejor
# guardar la transferencia (se acumulan hasta 5): la opción de hacer dos
# cambios la semana próxima vale más que una mejora marginal hoy.
BANK_THRESHOLD = 2.0
MAX_BANKED_TRANSFERS = 5
# Si el jugador que entra no sería titular hoy, su xP no se cobra mientras
# esté en el banco — solo entra por una auto-suplencia ocasional (si un
# titular no juega) o si más adelante, dentro del horizonte, pasa a
# titular por rotación/forma. No hay forma simple de estimar eso con
# precisión, así que en vez de contar el xP entero (optimista) o excluir
# el cambio directamente (visto en la práctica: el modelo cambió a Igor
# Jesus por Calvert-Lewin con un "+6.40 xP" que nunca se iba a cobrar
# porque Calvert-Lewin quedó en el banco), se descuenta la ganancia
# proyectada — el cambio sigue siendo posible si la ventaja alcanza para
# justificarlo igual, solo que el umbral efectivo sube.
BENCH_GAIN_DISCOUNT = 0.3
# Cuántas horas antes del deadline se decide la transferencia de la fecha.
# El gate natural sería "cambió el gameweek", pero eso dispara la decisión
# apenas termina la fecha anterior — el lunes, con la peor información de
# la semana. Las conferencias de prensa (donde se confirman lesiones) son
# jueves y viernes, y los precios se mueven todos los días. Esperar hasta
# el final de la ventana usa la mejor información disponible.
TRANSFER_DECISION_WINDOW_HOURS = 3


def _sell_price(now_cost: float, purchase_price: float) -> float:
    """Precio de venta real de FPL — no es simplemente `now_cost`.

    Si el jugador subió de precio desde que lo compraste, solo te quedás
    con la mitad de la ganancia (redondeada hacia abajo al escalón de
    £0.1m) — la otra mitad se pierde, es la regla oficial del juego. Si
    bajó o quedó igual, vendés al precio actual completo, sin descuento
    extra (el loss no se comparte).

    Ej.: comprado en £5.0m, ahora vale £5.3m (3 escalones de ganancia) →
    te quedás con 3//2=1 escalón → vendés en £5.1m, no en £5.3m.
    """
    if now_cost <= purchase_price:
        return now_cost
    profit_steps = round((now_cost - purchase_price) * 10)  # en escalones de £0.1m
    kept_steps = profit_steps // 2
    return round(purchase_price + kept_steps / 10, 1)


def load_my_team(path: str, players: pd.DataFrame) -> pd.DataFrame:
    """Lee el equipo del usuario (CSV con columna web_name y, opcional,
    team_name para desambiguar y purchase_price para el precio de venta
    real) y lo cruza contra el dataset de jugadores. Falla con un mensaje
    útil si un nombre no existe o es ambiguo.

    Si el CSV no trae `purchase_price` (o viene vacío para alguna fila),
    se asume que se compró al precio actual — sin ganancia ni pérdida
    todavía. Es el bootstrap razonable para un equipo recién armado; una
    vez que `evolve_base_squad`/`_apply_swap` hacen un cambio, sí queda
    el precio de compra real registrado.
    """
    team_file = pd.read_csv(path)
    if "web_name" not in team_file.columns:
        raise ValueError("El CSV del equipo necesita una columna 'web_name'.")

    matched = []
    purchase_prices = []
    for _, row in team_file.iterrows():
        name = str(row["web_name"]).strip()
        candidates = players[players["web_name"].str.lower() == name.lower()]
        if "team_name" in team_file.columns and pd.notna(row.get("team_name")):
            club = str(row["team_name"]).strip().lower()
            candidates = candidates[candidates["team_name"].str.lower() == club]

        if len(candidates) == 0:
            suggestions = difflib.get_close_matches(name, players["web_name"].tolist(), n=3)
            hint = f" ¿Quisiste decir: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"No encontré a '{name}' en el dataset.{hint}")
        if len(candidates) > 1:
            options = ", ".join(f"{r.web_name} ({r.team_name})" for r in candidates.itertuples())
            raise ValueError(
                f"'{name}' es ambiguo ({options}). Agregá la columna team_name para desambiguar."
            )
        matched.append(candidates.iloc[0])
        has_purchase_price = "purchase_price" in team_file.columns and pd.notna(row.get("purchase_price"))
        purchase_prices.append(float(row["purchase_price"]) if has_purchase_price else None)

    squad = pd.DataFrame(matched).reset_index(drop=True)
    squad["purchase_price"] = [
        pp if pp is not None else now_cost for pp, now_cost in zip(purchase_prices, squad["now_cost"])
    ]
    if len(squad) != 15:
        raise ValueError(f"El equipo tiene {len(squad)} jugadores, deben ser 15.")
    for position, count in SQUAD_COMPOSITION.items():
        actual = int((squad["position"] == position).sum())
        if actual != count:
            raise ValueError(f"Composición inválida: {actual} {position}, deben ser {count}.")
    if squad["team_name"].value_counts().max() > MAX_PER_CLUB:
        offender = squad["team_name"].value_counts().idxmax()
        raise ValueError(f"Más de {MAX_PER_CLUB} jugadores de {offender}.")
    return squad


def _candidate_pool(players: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    """Jugadores disponibles, con muestra confiable, y que no están ya en
    el equipo — los únicos que tiene sentido considerar para entrar.
    """
    return players[
        (players["status"] == "a")
        & (effective_minutes(players) >= MIN_MINUTES_FOR_RANKING)
        & (~players["id"].isin(squad["id"]))
    ]


def find_best_swaps(
    squad: pd.DataFrame, pool: pd.DataFrame, bank: float, top_n: int | None = None, stance: str = "neutral"
) -> pd.DataFrame:
    """Evalúa todos los cambios 1-por-1 legales (misma posición, entra en
    el presupuesto, respeta máx. 3 por club, y el que entra termina de
    TITULAR — ver abajo) y los ordena por ganancia de xP en el horizonte.
    `squad` debe ser SIEMPRE el equipo completo de 15 — el límite por club
    y la formación resultante se calculan sobre él.

    `stance` ("neutral"/"protect"/"chase"): con neutral (default), rankea
    por xp_horizon puro — igual que siempre. Con protect/chase, rankea por
    `rank_value` (ver add_rank_adjusted_value) calculado sobre squad+pool
    juntos, para que el percentil de ownership sea consistente entre quién
    sale y quién podría entrar. `out_xp`/`in_xp` en el resultado siempre
    muestran el xP crudo (sin ajustar) para que quede claro qué es medida
    objetiva y qué es la lente de la postura elegida.

    Presupuesto: usa el precio de venta REAL (`_sell_price`), no
    `now_cost` — `squad` necesita traer `purchase_price` (lo agrega
    `load_my_team`). Si el jugador subió de precio desde que lo
    compraste, solo recuperás la mitad de la ganancia al venderlo.
    """
    combined = add_rank_adjusted_value(
        pd.concat([squad, pool], ignore_index=True), stance, xp_column="xp_horizon"
    )
    value_by_id = combined.set_index("id")["rank_value"]

    # xP fecha por fecha, para poder medir la ganancia sobre el XI y no
    # sobre el jugador suelto (ver xi_horizon_gain). La postura se aplica
    # como factor por jugador —`rank_value` es `xp_horizon` multiplicado—
    # así que se traslada igual a cada fecha y protect/chase siguen
    # cambiando el ranking como antes.
    weekly = gameweek_expected_points(combined)
    stance_factor = (combined["rank_value"] / combined["xp_horizon"]).replace(
        [np.inf, -np.inf], 1.0
    ).fillna(1.0)
    weekly = weekly.mul(stance_factor, axis=0)
    weekly_by_id = dict(zip(combined["id"], weekly.to_numpy().tolist()))
    context = xi_gain_context(squad, weekly_by_id)

    club_counts = squad["team_name"].value_counts()
    swaps = []
    for out_player in squad.itertuples():
        sell_price = _sell_price(out_player.now_cost, out_player.purchase_price)
        budget = bank + sell_price
        candidates = pool[(pool["position"] == out_player.position) & (pool["now_cost"] <= budget)]
        for cand in candidates.itertuples():
            same_club_count = int(club_counts.get(cand.team_name, 0))
            if cand.team_name != out_player.team_name and same_club_count >= MAX_PER_CLUB:
                continue

            raw_gain = round(value_by_id[cand.id] - value_by_id[out_player.id], 2)
            if raw_gain <= 0:
                # Ya perdería contra el que sale aunque terminara de titular
                # — nunca va a ser el elegido, no vale la pena el chequeo
                # caro de abajo (rehacer la formación es lo más lento acá).
                continue

            # Ganancia real: cuánto sube el mejor XI de cada fecha, ya
            # contando al resto del plantel. Puede ser bastante menor que
            # `raw_gain` cuando el que sale iba a perder su lugar de todos
            # modos contra alguien que ya tenés — ver xi_horizon_gain.
            xi_gain, would_start, starts_any = xi_horizon_gain(
                context, out_player.id, out_player.position, weekly_by_id[cand.id]
            )
            # Si no arranca en ninguna fecha del horizonte, el XI no gana
            # nada y `xi_gain` es 0. Aun así el cambio no se descarta: se
            # cobra la fracción BENCH_GAIN_DISCOUNT del avance bruto, por
            # las auto-suplencias y por el calendario más allá del
            # horizonte. Es la misma decisión de siempre — penalización
            # suave, no prohibición.
            gain = xi_gain if starts_any else round(raw_gain * BENCH_GAIN_DISCOUNT, 2)

            swaps.append({
                "out_name": out_player.web_name,
                "out_club": out_player.team_name,
                "out_xp": out_player.xp_horizon,
                "in_name": cand.web_name,
                "in_club": cand.team_name,
                "in_xp": cand.xp_horizon,
                "in_cost": cand.now_cost,
                "in_fixtures": cand.fixtures_horizon,
                "gain": gain,
                "raw_gain": raw_gain,
                "would_start": would_start,
                "sell_price": sell_price,
                "cost_delta": round(cand.now_cost - sell_price, 1),
                "in_id": cand.id,
                "out_id": out_player.id,
            })
    if not swaps:
        return pd.DataFrame()
    result = pd.DataFrame(swaps).sort_values("gain", ascending=False).reset_index(drop=True)
    return result.head(top_n) if top_n is not None else result


def _apply_swap(squad: pd.DataFrame, pool: pd.DataFrame, swap: pd.Series) -> pd.DataFrame:
    # El que entra queda registrado como comprado a su precio actual —
    # necesario para calcular su propio precio de venta el día de mañana.
    incoming = pool[pool["id"] == swap["in_id"]].copy()
    incoming["purchase_price"] = incoming["now_cost"]
    return pd.concat([squad[squad["id"] != swap["out_id"]], incoming], ignore_index=True)


def _flag_problem_players(squad: pd.DataFrame) -> pd.DataFrame:
    """Jugadores del equipo con bandera: lesionados, suspendidos, o con
    probabilidad baja de jugar — los candidatos naturales a salir."""
    return squad[
        (squad["status"] != "a") | (squad["chance_of_playing_next_round"].fillna(100) < 100)
    ]


def _double_gameweek_note(horizon: int) -> str:
    fixtures = get_team_fixtures_horizon(horizon)
    per_gw = fixtures.groupby(["team_name", "gameweek"]).size()
    doubles = per_gw[per_gw > 1]
    if doubles.empty:
        return (
            f"No hay double gameweeks en las próximas {horizon} fechas — "
            "no es momento de Bench Boost ni Triple Captain (ver docs/chips-strategy.md)."
        )
    teams = ", ".join(f"{team} (GW{gw})" for (team, gw) in doubles.index)
    return f"¡Double gameweek detectado!: {teams}. Revisá docs/chips-strategy.md para el timing de chips."


def advise(team_path: str, bank: float = 0.0, free_transfers: int = 1, stance: str = "neutral") -> None:
    players = add_horizon_expected_points(add_expected_points(build_players_dataset()))
    squad = load_my_team(team_path, players)
    pool = _candidate_pool(players, squad)

    stance_note = {
        "neutral": "sin ajuste, solo xP",
        "protect": "favorece ownership alto (baja varianza)",
        "chase": "favorece diferenciales de ownership bajo (alta varianza)",
    }[stance]
    print(f"\n=== Asesor de transferencias — horizonte {HORIZON_GAMEWEEKS} fechas — postura: {stance} ({stance_note}) ===")
    print(f"Banco: £{bank:.1f}m | Transferencias libres: {free_transfers}")

    squad = squad.copy()
    squad["sell_price"] = squad.apply(lambda r: _sell_price(r["now_cost"], r["purchase_price"]), axis=1)
    market_value = squad["now_cost"].sum()
    spent = squad["purchase_price"].sum()
    sellable_value = squad["sell_price"].sum()
    print(f"Valor de mercado: £{market_value:.1f}m | Gastado: £{spent:.1f}m | "
          f"Recuperarías vendiendo todo: £{sellable_value:.1f}m + £{bank:.1f}m de banco = "
          f"£{sellable_value + bank:.1f}m")

    squad_cols = ["web_name", "team_name", "position", "purchase_price", "now_cost", "sell_price", "xp_horizon", "fixtures_horizon"]
    print("\n-- Tu equipo (ordenado por xP del horizonte) --")
    print(squad.sort_values("xp_horizon", ascending=False)[squad_cols].to_string(index=False))

    flagged = _flag_problem_players(squad)
    if not flagged.empty:
        print("\n-- ⚠ Jugadores con bandera --")
        for p in flagged.itertuples():
            chance = p.chance_of_playing_next_round
            chance_txt = f"{chance:.0f}% de jugar" if pd.notna(chance) else "sin probabilidad informada"
            print(f"  {p.web_name} ({p.team_name}) — status '{p.status}', {chance_txt}")

    all_swaps = find_best_swaps(squad, pool, bank, stance=stance)
    if all_swaps.empty:
        print("\nNo hay cambios legales que mejoren el equipo dentro del presupuesto.")
        return

    swaps = all_swaps.head(5)
    swap_cols = ["out_name", "in_name", "in_club", "gain", "would_start", "cost_delta", "in_fixtures"]
    print("\n-- Mejores cambios disponibles --")
    print(swaps[swap_cols].to_string(index=False))

    best = swaps.iloc[0]

    # Si hay un jugador con bandera y el mejor cambio no lo saca, mostramos
    # también la mejor forma de sacarlo — un jugador que no juega vale 0,
    # cualquier reemplazo sano suele ganarle a una mejora marginal.
    flagged_ids = set(flagged["id"])
    if flagged_ids and best["out_id"] not in flagged_ids:
        flagged_swaps = all_swaps[all_swaps["out_id"].isin(flagged_ids)]
        if not flagged_swaps.empty:
            fs = flagged_swaps.iloc[0]
            fs_note = "titular" if fs["would_start"] else "banco"
            print("\n-- Salida sugerida para el jugador con bandera --")
            print(f"  {fs['out_name']} → {fs['in_name']} ({fs['in_club']}, "
                  f"+{fs['gain']:.2f} xP, entra de {fs_note}, {fs['in_fixtures']})")

    print("\n-- Recomendación --")
    if free_transfers >= 1:
        if best["gain"] >= BANK_THRESHOLD or best["out_id"] in flagged_ids:
            best_note = "titular" if best["would_start"] else "banco"
            print(f"HACÉ EL CAMBIO: {best['out_name']} → {best['in_name']} "
                  f"(+{best['gain']:.2f} xP en {HORIZON_GAMEWEEKS} fechas, entra de {best_note}).")
        else:
            banked_next_week = min(free_transfers + 1, MAX_BANKED_TRANSFERS)
            print(f"GUARDÁ LA TRANSFERENCIA: la mejor mejora disponible es marginal "
                  f"(+{best['gain']:.2f} xP < umbral de {BANK_THRESHOLD}). "
                  f"La próxima fecha tendrías {banked_next_week} libres para un movimiento doble.")

        # ¿Y un segundo cambio? Greedy: aplicamos el mejor y re-evaluamos.
        squad_after = _apply_swap(squad, pool, best)
        pool_after = _candidate_pool(players, squad_after)
        bank_after = bank - best["cost_delta"]
        second_swaps = find_best_swaps(squad_after, pool_after, bank_after, top_n=1, stance=stance)
        if not second_swaps.empty:
            second = second_swaps.iloc[0]
            if free_transfers >= 2:
                if second["gain"] >= BANK_THRESHOLD:
                    print(f"SEGUNDO CAMBIO (libre): {second['out_name']} → {second['in_name']} "
                          f"(+{second['gain']:.2f} xP). Tenés {free_transfers} libres, usalo.")
                else:
                    print(f"El segundo cambio libre no vale la pena (+{second['gain']:.2f} xP) — guardalo.")
            else:
                net = second["gain"] - HIT_COST
                if second["gain"] > HIT_COST + HIT_UNCERTAINTY_MARGIN:
                    print(f"HIT DE -4 JUSTIFICADO: {second['out_name']} → {second['in_name']} "
                          f"gana +{second['gain']:.2f} xP (neto {net:+.2f} tras el hit, "
                          f"supera el margen de seguridad de {HIT_UNCERTAINTY_MARGIN}).")
                else:
                    print(f"NO PAGUES EL HIT: el mejor segundo cambio ({second['out_name']} → "
                          f"{second['in_name']}) gana +{second['gain']:.2f} xP — neto {net:+.2f} "
                          f"tras el -4, no supera el margen de seguridad.")

    # Capitanía: decisión de la próxima fecha (xp_next), no del horizonte —
    # se re-elige cada semana, no tiene sentido promediar 4 fechas.
    squad_captaincy = captaincy_picks(squad, top_n=2, stance=stance)
    if len(squad_captaincy) >= 2:
        cap, vice = squad_captaincy.iloc[0], squad_captaincy.iloc[1]
        print("\n-- Capitanía sugerida (próxima fecha) --")
        print(f"Capitán: {cap['web_name']} ({cap['xp_next']:.2f} xP vs {cap['next_opponent']}) | "
              f"Vice: {vice['web_name']} ({vice['xp_next']:.2f} xP vs {vice['next_opponent']}) — "
              f"el vice hereda la cinta si el capitán no juega")

    print(f"\n-- Chips --\n{_double_gameweek_note(HORIZON_GAMEWEEKS)}")


def _load_base_state() -> dict:
    if BASE_STATE_PATH.exists():
        return json.loads(BASE_STATE_PATH.read_text(encoding="utf-8"))
    # Antes de la primera evaluación: 0 transferencias acumuladas, para que
    # la primera fecha otorgue exactamente la 1 libre estándar de FPL.
    return {"last_evaluated_gameweek": None, "banked_free_transfers": 0}


def _save_base_state(state: dict) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BASE_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def plan_transfers(
    players: pd.DataFrame, squad: pd.DataFrame, free_transfers: int, gameweek: int
) -> tuple[pd.DataFrame, int, list[str]]:
    """Decide qué transferencias hacer y devuelve el plantel resultante,
    **sin persistir nada**.

    Es la lógica de decisión pura, compartida por dos usos que difieren
    solo en si el resultado se guarda: `evolve_base_squad` (lo aplica y lo
    persiste, una vez por fecha y dentro de la ventana del deadline) y
    `preview_base_transfers` (lo calcula a diario para anticipar
    contenido). Tenerla en un solo lugar evita que la propuesta que se
    publica y la decisión que se ejecuta puedan divergir.

    Devuelve (plantel, transferencias_libres_restantes, log).
    """
    pool = _candidate_pool(players, squad)
    flagged_ids = set(_flag_problem_players(squad)["id"])
    log: list[str] = []
    transfers_used = 0

    for _ in range(2):  # como mucho 2 movimientos por fecha, igual que advise()
        swaps = find_best_swaps(squad, pool, 0.0)
        if swaps.empty:
            break
        # Un jugador con bandera tiene prioridad de salida aunque no sea
        # el de mayor ganancia — puede valer 0 si no juega.
        flagged_swaps = swaps[swaps["out_id"].isin(flagged_ids)]
        candidate = flagged_swaps.iloc[0] if not flagged_swaps.empty else swaps.iloc[0]

        worth_free = candidate["gain"] >= BANK_THRESHOLD or candidate["out_id"] in flagged_ids
        worth_hit = candidate["gain"] > HIT_COST + HIT_UNCERTAINTY_MARGIN

        if free_transfers > 0 and worth_free:
            kind = "transferencia libre"
            free_transfers -= 1
        elif free_transfers == 0 and worth_hit:
            kind = "HIT -4 aplicado"
        else:
            break

        squad = _apply_swap(squad, pool, candidate)
        pool = _candidate_pool(players, squad)
        flagged_ids.discard(candidate["out_id"])
        transfers_used += 1
        starts_note = "titular" if candidate["would_start"] else "banco"
        log.append(
            f"{candidate['out_name']} → {candidate['in_name']} "
            f"(+{candidate['gain']:.2f} xP, {kind}, entra de {starts_note})"
        )

    if transfers_used == 0:
        log.append(
            f"Sin cambios en GW{gameweek} — transferencia guardada "
            f"({free_transfers} acumuladas para la próxima fecha)."
        )
    return squad, free_transfers, log


def preview_base_transfers(
    players: pd.DataFrame, team_path: str = "my_team.csv"
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Cómo quedaría el equipo Base si se aplicaran hoy las transferencias
    que el modelo propone — **sin tocar `my_team.csv` ni el estado**.

    Existe para poder preparar contenido con anticipación: la decisión
    real se toma dentro de las últimas horas antes del deadline (para
    aprovechar las lesiones ya confirmadas), pero un post se escribe
    antes. Esta vista se recalcula en cada corrida, así que refleja lo
    que el modelo haría con los datos de hoy.

    No es un compromiso: si el jueves se lesiona alguien, la decisión
    final del viernes puede ser otra.

    Devuelve (titulares, banco, log) del plantel proyectado.
    """
    state = _load_base_state()
    squad = load_my_team(team_path, players)
    current_gw = _current_gameweek(players)

    # Si la fecha ya se evaluó, `my_team.csv` YA tiene las transferencias
    # aplicadas: proponer más sería inventar movimientos que no existen.
    if state["last_evaluated_gameweek"] == current_gw:
        starters, bench = select_starting_xi(squad)
        return starters, bench, [f"GW{current_gw} ya ejecutado — el equipo Base ya incluye sus transferencias."]

    free_transfers = min(
        state["banked_free_transfers"] + _transfers_granted(current_gw), MAX_BANKED_TRANSFERS
    )
    squad, _, log = plan_transfers(players, squad, free_transfers, current_gw)
    starters, bench = select_starting_xi(squad)
    return starters, bench, log


def _transfers_granted(gameweek: int) -> int:
    """Transferencias libres que otorga el arranque de `gameweek`.

    GW1 no otorga ninguna: el equipo inicial se arma con cambios
    ilimitados hasta el deadline, así que no hay nada que "ahorrar". La
    primera transferencia libre de la temporada llega recién en GW2.

    Sin esto, el modelo se acreditaba una transferencia en GW1, no la
    usaba, y llegaba a GW2 creyendo tener dos — sugiriendo un movimiento
    doble que en la app real habría costado -4 puntos.
    """
    return 0 if gameweek <= 1 else 1


def _current_gameweek(players: pd.DataFrame) -> int:
    return int(players["next_gameweek"].dropna().mode().iloc[0])


def save_my_team(squad: pd.DataFrame, path: str) -> None:
    squad[["web_name", "team_name", "purchase_price"]].to_csv(path, index=False)


def evolve_base_squad(
    players: pd.DataFrame, team_path: str = "my_team.csv", force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Motor del equipo Base autogestionado: a diferencia de `advise` (que
    solo imprime una recomendación para que la leas), esto la EJECUTA.

    Dos frenos, no uno:

    1. **Una sola vez por gameweek** (estado en BASE_STATE_PATH): las
       transferencias de FPL son semanales, así que sin esto el equipo
       podría "gastar" una transferencia distinta cada día de la misma
       fecha, algo que en el juego no existe.
    2. **Solo dentro de las últimas `TRANSFER_DECISION_WINDOW_HOURS`
       antes del deadline**: el primer freno por sí solo dispara la
       decisión apenas termina la fecha anterior — el lunes, con la peor
       información de la semana. Las lesiones se confirman en las
       conferencias de jueves y viernes, y los precios cambian a diario.

    `force=True` saltea el segundo freno (no el primero). Sirve para
    correrlo a mano y ver qué haría, sin esperar a la ventana.

    Usa exactamente los mismos umbrales que `advise` (BANK_THRESHOLD,
    HIT_UNCERTAINTY_MARGIN) para decidir si mover, pero en vez de
    imprimir la decisión la aplica sobre `team_path` y sobre el conteo de
    transferencias acumuladas. Como se definió explícitamente: si el
    modelo considera que vale un hit de -4, lo aplica solo.

    Devuelve (starters, bench, log) — log es la lista de movimientos
    hechos en esta corrida (o el motivo por el que no se hizo ninguno).
    """
    state = _load_base_state()
    squad = load_my_team(team_path, players)
    current_gw = _current_gameweek(players)
    log = []

    deadline = get_next_deadline()
    if deadline is not None and not force:
        hours_left = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours_left > TRANSFER_DECISION_WINDOW_HOURS:
            log.append(
                f"Faltan {hours_left:.1f}h para el deadline de GW{current_gw} — se decide "
                f"dentro de las últimas {TRANSFER_DECISION_WINDOW_HOURS}h, cuando ya se "
                "conocen las lesiones reportadas."
            )
            starters, bench = select_starting_xi(squad)
            return starters, bench, log

    if state["last_evaluated_gameweek"] == current_gw:
        log.append(f"GW{current_gw} ya evaluado — sin cambios nuevos hasta la próxima fecha.")
        starters, bench = select_starting_xi(squad)
        return starters, bench, log

    free_transfers = min(
        state["banked_free_transfers"] + _transfers_granted(current_gw), MAX_BANKED_TRANSFERS
    )
    squad, free_transfers, plan_log = plan_transfers(players, squad, free_transfers, current_gw)
    log.extend(plan_log)

    save_my_team(squad, team_path)
    state["last_evaluated_gameweek"] = current_gw
    state["banked_free_transfers"] = free_transfers
    # Se persiste qué se movió, no solo el equipo resultante: mirando
    # `my_team.csv` o el CSV de equipos no hay forma de saber si un
    # jugador entró esta semana o llevaba meses. El briefing lo publica
    # para que la transferencia sea visible sin leer la salida del script.
    state["last_transfer_gameweek"] = current_gw
    state["last_transfers"] = log.copy()
    _save_base_state(state)

    starters, bench = select_starting_xi(squad)
    return starters, bench, log


def simulate_squad_trajectory(
    players: pd.DataFrame, weeks: int = HORIZON_GAMEWEEKS, team_path: str = "my_team.csv"
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Vista ESPECULATIVA de cómo evolucionaría el equipo Base en las
    próximas `weeks` fechas si el modelo corriera hoy mismo, semana a
    semana, la misma lógica de `evolve_base_squad` — sin esperar datos
    nuevos entre medio. No es una predicción ni se escribe en piedra: usa
    el `xp_horizon` de hoy para las decisiones de las 4 fechas; en la
    vida real, cada semana va a traer datos frescos (lesiones, precios,
    forma) que probablemente cambien el resultado real. Sirve para
    ilustrar "el plan de hoy, si nada cambiara" — no para comprometerse
    con él.

    A diferencia de evolve_base_squad, esto es de **solo lectura**: nunca
    toca `my_team.csv` ni el estado persistido — trabaja sobre copias.

    **Cada columna es el plantel que JUEGA esa fecha**, o sea con la
    decisión de esa fecha ya aplicada. Eso incluye la primera columna: si
    el modelo propone una transferencia para la fecha en curso, la columna
    de hoy ya la muestra, igual que `squad_type = "Base proyectado"` en
    `squad_recommendations.csv`.

    Antes el loop arrancaba en la segunda fecha y sembraba la primera
    columna con `my_team.csv` tal cual, así que la transferencia de esta
    fecha aparecía recién en la columna siguiente: la vista contradecía al
    Base proyectado con una fecha de desfase. La excepción es cuando la
    fecha en curso ya se ejecutó — ahí `my_team.csv` YA tiene sus
    transferencias y proponer otra sería inventar un movimiento, el mismo
    criterio que usa `preview_base_transfers`.

    Devuelve (trayectoria, notas):
    - trayectoria: 15 filas (una por slot del plantel, con el slot fijo a
      lo largo de las columnas), columnas `position`,
      `GW{n}`...`GW{n+weeks-1}` con el nombre del jugador en ese slot esa
      fecha. Si no hubo cambio en la semana, el nombre se repite.
    - notas: un dict {gameweek: descripción de los cambios de esa semana}.
    """
    state = _load_base_state()
    squad = load_my_team(team_path, players)
    free_transfers = state["banked_free_transfers"]
    current_gw = _current_gameweek(players)

    slots = squad[["id", "web_name", "position"]].to_dict("records")
    positions = [slot["position"] for slot in slots]
    trajectory: dict[str, list[str]] = {}
    notes: dict[str, str] = {}

    for offset in range(weeks):
        gameweek = current_gw + offset
        label = f"GW{gameweek}"

        if offset == 0 and state["last_evaluated_gameweek"] == current_gw:
            notes[label] = f"GW{gameweek} ya ejecutado — my_team.csv ya incluye sus transferencias."
            trajectory[label] = [slot["web_name"] for slot in slots]
            continue

        free_transfers = min(
            free_transfers + _transfers_granted(gameweek), MAX_BANKED_TRANSFERS
        )
        previous_ids = set(squad["id"])
        # Se reusa el mismo núcleo de decisión que ejecuta el equipo real:
        # si esta simulación tuviera su propia copia de la lógica, podría
        # anunciar un plan que el modelo nunca haría.
        squad, free_transfers, log = plan_transfers(players, squad, free_transfers, gameweek)
        _reassign_slots(slots, squad, previous_ids)

        trajectory[label] = [slot["web_name"] for slot in slots]
        notes[label] = "; ".join(log)

    result = pd.DataFrame(trajectory)
    result.insert(0, "position", positions)
    return result, notes


def _reassign_slots(slots: list[dict], squad: pd.DataFrame, previous_ids: set) -> None:
    """Mantiene fijo el "slot" de cada jugador entre columnas de la
    trayectoria, mutando `slots` en el lugar.

    Hace falta porque `_apply_swap` no conserva el orden de las filas
    (saca al que se va y agrega al que entra al final), y la tabla de
    trayectoria se lee en horizontal: la gracia es ver qué nombre ocupa
    la misma línea fecha a fecha. Cada jugador que entra se asigna al slot
    de alguien que salió en su misma posición — los cambios siempre son
    dentro de la misma posición, así que el emparejamiento existe. Si dos
    salidas de la misma posición caen en la misma fecha, da igual cuál de
    los dos slots recibe a cuál: la fila sigue siendo esa posición.
    """
    vacated = previous_ids - set(squad["id"])
    for incoming in squad[~squad["id"].isin(previous_ids)].itertuples():
        index = next(
            i for i, slot in enumerate(slots)
            if slot["id"] in vacated and slot["position"] == incoming.position
        )
        vacated.discard(slots[index]["id"])
        slots[index] = {
            "id": incoming.id,
            "web_name": incoming.web_name,
            "position": incoming.position,
        }
