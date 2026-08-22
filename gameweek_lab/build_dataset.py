import json
from datetime import datetime

import pandas as pd

from gameweek_lab.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, PLAYER_PHOTO_URL_TEMPLATE


def _load_raw(name: str):
    path = DATA_RAW_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _next_gameweek(fixtures: list[dict]) -> int:
    upcoming_events = sorted({f["event"] for f in fixtures if not f["finished"] and f["event"] is not None})
    return upcoming_events[0]


def _next_fixture_by_team(fixtures: list[dict]) -> dict[int, dict]:
    """Para cada equipo, su próximo fixture sin jugar (el de gameweek más bajo)."""
    upcoming = [f for f in fixtures if not f["finished"] and f["event"] is not None]
    upcoming.sort(key=lambda f: f["event"])

    next_fixture: dict[int, dict] = {}
    for fixture in upcoming:
        for team_id, is_home in ((fixture["team_h"], True), (fixture["team_a"], False)):
            if team_id in next_fixture:
                continue  # ya guardamos el próximo fixture de este equipo, era el más temprano
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            next_fixture[team_id] = {
                "gameweek": fixture["event"],
                "opponent_id": opponent_id,
                "is_home": is_home,
                "difficulty": difficulty,
            }
    return next_fixture


def get_next_gameweek_fixtures() -> pd.DataFrame:
    """Los partidos del próximo gameweek: quién juega contra quién.

    La usa squad_builder para evitar que el plantel tenga, a la vez, al
    defensor/arquero de un equipo y a un mediocampista/delantero del rival
    que enfrenta ese mismo gameweek — un jugador propio le rompería el
    clean sheet al otro.
    """
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")
    team_names = pd.DataFrame(bootstrap["teams"]).set_index("id")["name"]

    next_gw = _next_gameweek(fixtures)
    rows = [
        {
            "gameweek": f["event"],
            "team_h_name": team_names[f["team_h"]],
            "team_a_name": team_names[f["team_a"]],
        }
        for f in fixtures
        if f["event"] == next_gw
    ]
    return pd.DataFrame(rows)


def get_team_fixtures_horizon(horizon: int = 4) -> pd.DataFrame:
    """Todos los fixtures de cada equipo en las próximas `horizon` fechas,
    una fila por (equipo, partido).

    A diferencia de _next_fixture_by_team (que devuelve solo el próximo
    partido), acá un equipo puede aparecer dos veces en la misma fecha
    (double gameweek) o ninguna (blank gameweek) — el xP a horizonte
    simplemente suma lo que haya, así esos casos quedan bien contados sin
    lógica especial.
    """
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")
    teams = pd.DataFrame(bootstrap["teams"]).set_index("id")

    upcoming_events = sorted({f["event"] for f in fixtures if not f["finished"] and f["event"] is not None})
    target_gws = set(upcoming_events[:horizon])

    rows = []
    for fixture in fixtures:
        if fixture["event"] not in target_gws:
            continue
        for team_id, is_home in ((fixture["team_h"], True), (fixture["team_a"], False)):
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            rows.append({
                "team_name": teams.loc[team_id, "name"],
                "gameweek": fixture["event"],
                "opponent": teams.loc[opponent_id, "short_name"],
                "is_home": is_home,
                "difficulty": difficulty,
            })
    return pd.DataFrame(rows)


def _set_piece_duties(players: pd.DataFrame) -> pd.Series:
    """Resumen legible de qué jugadas a balón parado ejecuta el jugador,
    p. ej. "Penales, Córners". Vacío si no es primera opción en ninguna.

    Solo se considera el orden 1 (primera opción): el segundo de la lista
    patea tan pocas veces que no cambia una decisión.

    Es información de CONTEXTO, no entra al cálculo de xP — ver el
    comentario extenso en el README sobre el riesgo de doble conteo.
    """
    duties = {
        "penalties_order": "Penales",
        "direct_freekicks_order": "Tiros libres",
        "corners_and_indirect_freekicks_order": "Córners",
    }
    labels = pd.Series([[] for _ in range(len(players))], index=players.index)
    for column, label in duties.items():
        is_first = players[column] == 1
        labels[is_first] = labels[is_first].map(lambda existing, l=label: existing + [l])
    return labels.map(", ".join)


def get_next_deadline() -> datetime | None:
    """Fecha/hora límite del próximo gameweek, en UTC.

    La usa `evolve_base_squad` para decidir transferencias cerca del
    deadline y no apenas termina la fecha anterior — ver
    TRANSFER_DECISION_WINDOW_HOURS.
    """
    bootstrap = _load_raw("bootstrap-static")
    upcoming = [e for e in bootstrap["events"] if not e["finished"]]
    if not upcoming:
        return None
    next_event = min(upcoming, key=lambda e: e["deadline_time"])
    return datetime.fromisoformat(next_event["deadline_time"].replace("Z", "+00:00"))


def get_matches_played_by_team() -> dict[str, int]:
    """Cuántos partidos lleva jugados cada equipo — el denominador para
    calcular con qué frecuencia un jugador es titular (`starts` es un
    acumulado, no una tasa).

    Se cuenta desde `fixtures` (no desde los `events` de bootstrap) para
    que un equipo con partidos postergados quede con su cuenta real, no
    con la del calendario nominal.

    Cuenta por `started`, no por `finished`: FPL marca `finished` recién
    cuando termina de procesar los datos del partido, pero los acumulados
    del jugador (minutos, starts, xG) ya se actualizan apenas se juega.
    Usar `finished` dejaba una ventana — verificada durante GW1, con 6
    partidos jugados y 0 marcados como terminados — en la que el modelo
    creía estar en pre-temporada mientras los datos ya eran de la
    temporada nueva.

    Pre-temporada devuelve 0 para todos: ahí `starts` viene de la
    temporada anterior, y quien llama debe usar `FULL_SEASON_MATCHES`
    como denominador (ver `_start_rate` en analysis.py).
    """
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")
    team_names = pd.DataFrame(bootstrap["teams"]).set_index("id")["name"]

    played = {name: 0 for name in team_names}
    for fixture in fixtures:
        if not (fixture.get("started") or fixture["finished"]):
            continue
        for team_id in (fixture["team_h"], fixture["team_a"]):
            played[team_names[team_id]] += 1
    return played


def build_players_dataset() -> pd.DataFrame:
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"]).set_index("id")
    positions = pd.DataFrame(bootstrap["element_types"]).set_index("id")

    # La API manda varios números como texto (ej. "4.4") — hay que convertirlos
    numeric_as_text = ["form", "points_per_game", "selected_by_percent", "ep_next", "ep_this"]
    for col in numeric_as_text:
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players["now_cost"] = players["now_cost"] / 10.0  # el precio viene x10 (60 = £6.0m)

    players["team_name"] = players["team"].map(teams["name"])
    players["position"] = players["element_type"].map(positions["singular_name_short"])
    players["full_name"] = players["first_name"] + " " + players["second_name"]
    players["photo_url"] = players["code"].map(lambda code: PLAYER_PHOTO_URL_TEMPLATE.format(code=code))
    players["set_piece_duties"] = _set_piece_duties(players)

    next_fixture = _next_fixture_by_team(fixtures)
    players["next_gameweek"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("gameweek"))
    players["next_opponent"] = players["team"].map(
        lambda t: teams["name"].get(next_fixture.get(t, {}).get("opponent_id"))
    )
    players["next_is_home"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("is_home"))
    players["next_fixture_difficulty"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("difficulty"))

    columns = [
        "id", "web_name", "full_name", "team_name", "position", "now_cost", "status", "photo_url",
        "total_points", "points_per_game", "form", "selected_by_percent",
        "minutes", "starts", "chance_of_playing_next_round", "ep_next", "set_piece_duties",
        "expected_goals_per_90", "expected_assists_per_90", "expected_goals_conceded_per_90",
        # Momentum de transferencias — para anticipar subidas/bajadas de
        # precio. Pre-temporada están en 0 para todos (armar tu plantel
        # inicial no cuenta como "transferencia" en FPL); solo van a tener
        # señal real una vez que arranque la temporada.
        "transfers_in_event", "transfers_out_event",
        "next_gameweek", "next_opponent", "next_is_home", "next_fixture_difficulty",
    ]
    players = players[columns].sort_values("total_points", ascending=False).reset_index(drop=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "players.csv"
    players.to_csv(out_path, index=False)
    print(f"Guardado {out_path} ({len(players)} jugadores)")
    return players


if __name__ == "__main__":
    build_players_dataset()
