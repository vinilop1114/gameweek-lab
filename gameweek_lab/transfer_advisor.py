import difflib
import json

import pandas as pd

from gameweek_lab.analysis import (
    HORIZON_GAMEWEEKS,
    MIN_MINUTES_FOR_RANKING,
    add_expected_points,
    add_horizon_expected_points,
)
from gameweek_lab.build_dataset import build_players_dataset, get_team_fixtures_horizon
from gameweek_lab.config import DATA_PROCESSED_DIR
from gameweek_lab.squad_builder import MAX_PER_CLUB, SQUAD_COMPOSITION, select_starting_xi

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


def load_my_team(path: str, players: pd.DataFrame) -> pd.DataFrame:
    """Lee el equipo del usuario (CSV con columna web_name y, opcional,
    team_name para desambiguar) y lo cruza contra el dataset de jugadores.
    Falla con un mensaje útil si un nombre no existe o es ambiguo.
    """
    team_file = pd.read_csv(path)
    if "web_name" not in team_file.columns:
        raise ValueError("El CSV del equipo necesita una columna 'web_name'.")

    matched = []
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

    squad = pd.DataFrame(matched).reset_index(drop=True)
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
        & (players["minutes"] >= MIN_MINUTES_FOR_RANKING)
        & (~players["id"].isin(squad["id"]))
    ]


def find_best_swaps(squad: pd.DataFrame, pool: pd.DataFrame, bank: float, top_n: int | None = None) -> pd.DataFrame:
    """Evalúa todos los cambios 1-por-1 legales (misma posición, entra en
    el presupuesto, respeta máx. 3 por club) y los ordena por ganancia de
    xP en el horizonte. `squad` debe ser SIEMPRE el equipo completo de 15 —
    el límite por club se calcula sobre él.

    Simplificación: asumimos que el jugador se vende a su precio actual.
    FPL en realidad paga el precio de compra + la mitad de la subida, pero
    la API no expone tu precio de compra — para diferencias de ±0.1-0.2m
    el ranking de cambios casi nunca cambia.
    """
    club_counts = squad["team_name"].value_counts()
    swaps = []
    for out_player in squad.itertuples():
        budget = bank + out_player.now_cost
        candidates = pool[(pool["position"] == out_player.position) & (pool["now_cost"] <= budget)]
        for cand in candidates.itertuples():
            same_club_count = int(club_counts.get(cand.team_name, 0))
            if cand.team_name != out_player.team_name and same_club_count >= MAX_PER_CLUB:
                continue
            swaps.append({
                "out_name": out_player.web_name,
                "out_club": out_player.team_name,
                "out_xp": out_player.xp_horizon,
                "in_name": cand.web_name,
                "in_club": cand.team_name,
                "in_xp": cand.xp_horizon,
                "in_cost": cand.now_cost,
                "in_fixtures": cand.fixtures_horizon,
                "gain": round(cand.xp_horizon - out_player.xp_horizon, 2),
                "cost_delta": round(cand.now_cost - out_player.now_cost, 1),
                "in_id": cand.id,
                "out_id": out_player.id,
            })
    if not swaps:
        return pd.DataFrame()
    result = pd.DataFrame(swaps).sort_values("gain", ascending=False).reset_index(drop=True)
    return result.head(top_n) if top_n is not None else result


def _apply_swap(squad: pd.DataFrame, pool: pd.DataFrame, swap: pd.Series) -> pd.DataFrame:
    incoming = pool[pool["id"] == swap["in_id"]]
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


def advise(team_path: str, bank: float = 0.0, free_transfers: int = 1) -> None:
    players = add_horizon_expected_points(add_expected_points(build_players_dataset()))
    squad = load_my_team(team_path, players)
    pool = _candidate_pool(players, squad)

    print(f"\n=== Asesor de transferencias — horizonte {HORIZON_GAMEWEEKS} fechas ===")
    print(f"Banco: £{bank:.1f}m | Transferencias libres: {free_transfers}")

    squad_cols = ["web_name", "team_name", "position", "now_cost", "xp_horizon", "fixtures_horizon"]
    print("\n-- Tu equipo (ordenado por xP del horizonte) --")
    print(squad.sort_values("xp_horizon", ascending=False)[squad_cols].to_string(index=False))

    flagged = _flag_problem_players(squad)
    if not flagged.empty:
        print("\n-- ⚠ Jugadores con bandera --")
        for p in flagged.itertuples():
            chance = p.chance_of_playing_next_round
            chance_txt = f"{chance:.0f}% de jugar" if pd.notna(chance) else "sin probabilidad informada"
            print(f"  {p.web_name} ({p.team_name}) — status '{p.status}', {chance_txt}")

    all_swaps = find_best_swaps(squad, pool, bank)
    if all_swaps.empty:
        print("\nNo hay cambios legales que mejoren el equipo dentro del presupuesto.")
        return

    swaps = all_swaps.head(5)
    swap_cols = ["out_name", "in_name", "in_club", "gain", "cost_delta", "in_fixtures"]
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
            print("\n-- Salida sugerida para el jugador con bandera --")
            print(f"  {fs['out_name']} → {fs['in_name']} ({fs['in_club']}, "
                  f"+{fs['gain']:.2f} xP, {fs['in_fixtures']})")

    print("\n-- Recomendación --")
    if free_transfers >= 1:
        if best["gain"] >= BANK_THRESHOLD or best["out_id"] in flagged_ids:
            print(f"HACÉ EL CAMBIO: {best['out_name']} → {best['in_name']} "
                  f"(+{best['gain']:.2f} xP en {HORIZON_GAMEWEEKS} fechas).")
        else:
            banked_next_week = min(free_transfers + 1, MAX_BANKED_TRANSFERS)
            print(f"GUARDÁ LA TRANSFERENCIA: la mejor mejora disponible es marginal "
                  f"(+{best['gain']:.2f} xP < umbral de {BANK_THRESHOLD}). "
                  f"La próxima fecha tendrías {banked_next_week} libres para un movimiento doble.")

        # ¿Y un segundo cambio? Greedy: aplicamos el mejor y re-evaluamos.
        squad_after = _apply_swap(squad, pool, best)
        pool_after = _candidate_pool(players, squad_after)
        bank_after = bank - best["cost_delta"]
        second_swaps = find_best_swaps(squad_after, pool_after, bank_after, top_n=1)
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
    likely_starters = squad[
        squad["chance_of_playing_next_round"].fillna(100) >= 75
    ].sort_values("xp_next", ascending=False)
    if len(likely_starters) >= 2:
        cap, vice = likely_starters.iloc[0], likely_starters.iloc[1]
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


def _current_gameweek(players: pd.DataFrame) -> int:
    return int(players["next_gameweek"].dropna().mode().iloc[0])


def save_my_team(squad: pd.DataFrame, path: str) -> None:
    squad[["web_name", "team_name"]].to_csv(path, index=False)


def evolve_base_squad(players: pd.DataFrame, team_path: str = "my_team.csv") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Motor del equipo Base autogestionado: a diferencia de `advise` (que
    solo imprime una recomendación para que la leas), esto la EJECUTA.

    Se evalúa una sola vez por gameweek, no una vez por corrida diaria —
    las transferencias de FPL son semanales, así que sin este freno el
    equipo podría "gastar" una transferencia distinta cada día dentro de
    la misma fecha, algo que en el juego real no existe. El freno es el
    estado persistido en BASE_STATE_PATH (última fecha evaluada).

    Usa exactamente los mismos umbrales que `advise` (BANK_THRESHOLD,
    HIT_UNCERTAINTY_MARGIN) para decidir si mover, pero en vez de
    imprimir la decisión la aplica sobre `team_path` y sobre el conteo de
    transferencias acumuladas. Como se definió explícitamente: si el
    modelo considera que vale un hit de -4, lo aplica solo.

    Devuelve (starters, bench, log) — log es la lista de movimientos
    hechos en esta corrida (o un aviso de que ya se evaluó esta fecha).
    """
    state = _load_base_state()
    squad = load_my_team(team_path, players)
    current_gw = _current_gameweek(players)
    log = []

    if state["last_evaluated_gameweek"] == current_gw:
        log.append(f"GW{current_gw} ya evaluado — sin cambios nuevos hasta la próxima fecha.")
        starters, bench = select_starting_xi(squad)
        return starters, bench, log

    # +1 transferencia libre al arrancar la fecha, tope de 5 acumuladas.
    free_transfers = min(state["banked_free_transfers"] + 1, MAX_BANKED_TRANSFERS)
    pool = _candidate_pool(players, squad)
    flagged_ids = set(_flag_problem_players(squad)["id"])
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
            squad = _apply_swap(squad, pool, candidate)
            pool = _candidate_pool(players, squad)
            flagged_ids.discard(candidate["out_id"])
            free_transfers -= 1
            transfers_used += 1
            log.append(f"{candidate['out_name']} → {candidate['in_name']} "
                       f"(+{candidate['gain']:.2f} xP, transferencia libre)")
        elif free_transfers == 0 and worth_hit:
            squad = _apply_swap(squad, pool, candidate)
            pool = _candidate_pool(players, squad)
            flagged_ids.discard(candidate["out_id"])
            transfers_used += 1
            log.append(f"{candidate['out_name']} → {candidate['in_name']} "
                       f"(+{candidate['gain']:.2f} xP, HIT -4 aplicado)")
        else:
            break

    if transfers_used == 0:
        log.append(f"Sin cambios en GW{current_gw} — transferencia guardada "
                   f"({free_transfers} acumuladas para la próxima fecha).")

    save_my_team(squad, team_path)
    state["last_evaluated_gameweek"] = current_gw
    state["banked_free_transfers"] = free_transfers
    _save_base_state(state)

    starters, bench = select_starting_xi(squad)
    return starters, bench, log
