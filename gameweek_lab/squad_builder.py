import pandas as pd
import pulp

from gameweek_lab.analysis import MIN_MINUTES_FOR_RANKING, add_expected_points
from gameweek_lab.build_dataset import build_players_dataset, get_next_gameweek_fixtures
from gameweek_lab.config import DATA_PROCESSED_DIR

BUDGET = 100.0
SQUAD_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3

# Formaciones válidas para los 11 titulares: 1 GK fijo + 10 de campo,
# con al menos 3 DEF, 2 MID, 1 FWD (regla oficial de FPL).
STARTING_FORMATIONS = [
    (d, m, f)
    for d in range(3, 6)
    for m in range(2, 6)
    for f in range(1, 4)
    if d + m + f == 10
]

# Peso del xP de los titulares frente al costo del plantel en el objetivo
# del Wildcard. Tiene que ser lo bastante grande para que ganar 0.01 de xP
# en el XI siempre valga más que cualquier ahorro posible en el banco
# (el ahorro máximo posible es del orden del presupuesto total, £100m).
WILDCARD_XI_WEIGHT = 100_000

# Costo estimado de un "enfrentamiento interno": tener al DEF/GK de un
# equipo y a un MID/FWD del rival que enfrenta esa misma fecha. Si el
# atacante anota, mata el clean sheet (4 pts) del propio defensor.
# Estimación gruesa: un buen atacante anota en ~35% de sus partidos, y el
# clean sheet estaba vivo hasta ese gol en ~40% de los casos:
# 0.35 × 4 × 0.40 ≈ 0.5 xP. No es una prohibición — una dupla que gane
# más que esto sigue entrando al equipo, pagando su precio.
CLASH_PENALTY_XP = 0.5


def _eligible_players(players: pd.DataFrame) -> pd.DataFrame:
    """Jugadores disponibles y con muestra confiable (ver MIN_MINUTES_FOR_RANKING
    en analysis.py) — el punto de partida para cualquier optimización de plantel.
    """
    return players[
        (players["status"] == "a") & (players["minutes"] >= MIN_MINUTES_FOR_RANKING)
    ].reset_index(drop=True)


def _add_position_quota_constraints(problem: pulp.LpProblem, pick: dict, available: pd.DataFrame) -> None:
    for position, count in SQUAD_COMPOSITION.items():
        idx = available.index[available["position"] == position]
        problem += pulp.lpSum(pick[i] for i in idx) == count


def _add_club_limit_constraints(problem: pulp.LpProblem, pick: dict, available: pd.DataFrame) -> None:
    for team in available["team_name"].unique():
        idx = available.index[available["team_name"] == team]
        problem += pulp.lpSum(pick[i] for i in idx) <= MAX_PER_CLUB


def _internal_clash_penalty(problem: pulp.LpProblem, pick: dict, available: pd.DataFrame):
    """Le pone precio (no prohibición) a los enfrentamientos internos:
    tener al defensor/arquero de un equipo y a un mediocampista/delantero
    del rival que enfrenta ese mismo gameweek.

    Antes esto era una restricción dura (pick_d + pick_a <= 1). Ahora es
    una penalización de CLASH_PENALTY_XP en el objetivo: si una dupla que
    se enfrenta entre sí proyecta ganar más que ese costo, el solver la
    elige igual — el auto-sabotaje está permitido cuando los números lo
    justifican.

    Técnica: por cada par en conflicto se crea una variable binaria z con
    la restricción z >= pick_d + pick_a - 1. Como z se resta del objetivo
    (que se maximiza), el solver la deja en 0 salvo cuando ambos jugadores
    están elegidos, donde la restricción la fuerza a 1 y el par paga su
    penalización.
    """
    fixtures = get_next_gameweek_fixtures()
    defensive_positions = ["GKP", "DEF"]
    attacking_positions = ["MID", "FWD"]

    clash_vars = []
    for fixture_i, fixture in fixtures.iterrows():
        team_a, team_b = fixture["team_h_name"], fixture["team_a_name"]
        for defense_team, attack_team in ((team_a, team_b), (team_b, team_a)):
            defenders = available.index[
                (available["team_name"] == defense_team) & (available["position"].isin(defensive_positions))
            ]
            attackers = available.index[
                (available["team_name"] == attack_team) & (available["position"].isin(attacking_positions))
            ]
            for d in defenders:
                for a in attackers:
                    z = pulp.LpVariable(f"clash_{fixture_i}_{d}_{a}", cat="Binary")
                    problem += z >= pick[d] + pick[a] - 1
                    clash_vars.append(z)
    return CLASH_PENALTY_XP * pulp.lpSum(clash_vars)


def select_squad(players: pd.DataFrame) -> pd.DataFrame:
    """Elige los 15 jugadores que maximizan el xP total, respetando
    presupuesto, composición de posiciones y máximo 3 por club. Los
    enfrentamientos internos están permitidos pero pagan su costo
    esperado (ver _internal_clash_penalty).

    Pensado como equipo base de la temporada: el presupuesto se reparte
    entre los 15, así que el banco también queda jugable (importa, porque
    lo vas a sostener varias fechas, no solo esta).
    """
    available = _eligible_players(players)

    problem = pulp.LpProblem("squad_selection", pulp.LpMaximize)
    pick = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in available.index}

    clash_penalty = _internal_clash_penalty(problem, pick, available)
    problem += (
        pulp.lpSum(pick[i] * available.loc[i, "xp_next"] for i in available.index) - clash_penalty
    )
    problem += pulp.lpSum(pick[i] * available.loc[i, "now_cost"] for i in available.index) <= BUDGET

    _add_position_quota_constraints(problem, pick, available)
    _add_club_limit_constraints(problem, pick, available)

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"No se encontró una solución óptima (status: {status})")

    selected_idx = [i for i in available.index if pick[i].value() == 1]
    return available.loc[selected_idx].reset_index(drop=True)


def select_starting_xi(squad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """De los 15 del plantel, elige los 11 que arrancan probando cada
    formación válida y quedándose con la de mayor xP total.

    Con 15 jugadores fijos y ~13 formaciones posibles, no hace falta un
    solver: alcanza con probar todas y comparar. Es fuerza bruta, pero
    exacta, porque el espacio de búsqueda es chico y ya está acotado por
    la selección del plantel.
    """
    by_position = {
        pos: squad[squad["position"] == pos].sort_values("xp_next", ascending=False)
        for pos in ["GKP", "DEF", "MID", "FWD"]
    }
    goalkeeper = by_position["GKP"].head(1)

    best_total = -1.0
    best_formation = STARTING_FORMATIONS[0]
    for d, m, f in STARTING_FORMATIONS:
        total = (
            by_position["DEF"].head(d)["xp_next"].sum()
            + by_position["MID"].head(m)["xp_next"].sum()
            + by_position["FWD"].head(f)["xp_next"].sum()
        )
        if total > best_total:
            best_total = total
            best_formation = (d, m, f)

    d, m, f = best_formation
    starters = pd.concat([
        goalkeeper,
        by_position["DEF"].head(d),
        by_position["MID"].head(m),
        by_position["FWD"].head(f),
    ])

    bench = _order_bench(squad, starters)
    return starters.sort_values("xp_next", ascending=False), bench


def _order_bench(squad: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Ordena el banco según la mecánica real de FPL:

    - Slot 1: el arquero suplente, fijo — FPL solo intercambia arqueros
      entre sí, así que su posición en el orden no compite con nadie.
    - Slots 2-4: los de campo, por xP descendente. Poner al de mayor xP
      primero no arriesga nada: cuando un titular no juega, FPL recorre el
      banco en orden y saltea automáticamente a quien rompería la
      formación mínima (>=3 DEF, >=2 MID, >=1 FWD) — así que priorizar al
      mejor puntuador esperado es óptimo, la validez la garantiza la regla.
    """
    bench = squad[~squad["id"].isin(starters["id"])]
    bench_gk = bench[bench["position"] == "GKP"]
    bench_outfield = bench[bench["position"] != "GKP"].sort_values("xp_next", ascending=False)
    bench = pd.concat([bench_gk, bench_outfield]).reset_index(drop=True)
    bench["bench_order"] = range(1, len(bench) + 1)
    return bench


def _first_valid_sub(missing, starters: pd.DataFrame, bench: pd.DataFrame) -> str | None:
    """Simula la auto-suplencia de FPL: si `missing` no juega, devuelve el
    nombre del primer suplente (en orden de banco) cuyo ingreso mantiene
    una formación válida. Arquero solo por arquero; para los de campo se
    verifica el mínimo por posición del XI resultante.
    """
    if missing.position == "GKP":
        bench_gk = bench[bench["position"] == "GKP"]
        return bench_gk.iloc[0]["web_name"] if len(bench_gk) else None

    remaining_counts = starters[starters["id"] != missing.id]["position"].value_counts()
    for sub in bench.itertuples():
        if sub.position == "GKP":
            continue
        defenders = remaining_counts.get("DEF", 0) + (sub.position == "DEF")
        midfielders = remaining_counts.get("MID", 0) + (sub.position == "MID")
        forwards = remaining_counts.get("FWD", 0) + (sub.position == "FWD")
        if defenders >= 3 and midfielders >= 2 and forwards >= 1:
            return sub.web_name
    return None


def select_wildcard_squad(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arma plantel + 11 titulares asumiendo que se puede usar Wildcard
    todas las fechas: como el banco nunca se sostiene más de una semana,
    no vale la pena pagar por profundidad ahí. Gasta lo mínimo posible en
    los 4 suplentes y vuelca casi todo el presupuesto a los titulares.

    A diferencia de select_squad + select_starting_xi (dos pasos
    separados), acá el plantel y los titulares se deciden en un solo ILP.
    Si los separáramos, habría que "reservar" presupuesto para el banco
    antes de saber qué banco hace falta — circular. Resolviendo todo
    junto, el solver lo maneja sin ese problema: el peso grande sobre el
    xP de los titulares (WILDCARD_XI_WEIGHT) hace que maximizar el XI sea
    siempre la prioridad, y minimizar el costo total del plantel es el
    criterio de desempate que empuja el gasto del banco al mínimo.
    """
    available = _eligible_players(players)

    problem = pulp.LpProblem("wildcard_selection", pulp.LpMaximize)
    in_squad = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in available.index}
    in_xi = {i: pulp.LpVariable(f"xi_{i}", cat="Binary") for i in available.index}

    for i in available.index:
        problem += in_xi[i] <= in_squad[i]  # solo puede arrancar quien está en el plantel

    # La penalización por enfrentamientos internos solo mira a los
    # titulares: un suplente que no juega no le rompe el clean sheet a nadie.
    xi_clash_penalty = _internal_clash_penalty(problem, in_xi, available)
    problem += (
        WILDCARD_XI_WEIGHT
        * (
            pulp.lpSum(in_xi[i] * available.loc[i, "xp_next"] for i in available.index)
            - xi_clash_penalty
        )
        - pulp.lpSum(in_squad[i] * available.loc[i, "now_cost"] for i in available.index)
    )

    problem += pulp.lpSum(in_squad[i] * available.loc[i, "now_cost"] for i in available.index) <= BUDGET
    _add_position_quota_constraints(problem, in_squad, available)
    _add_club_limit_constraints(problem, in_squad, available)

    problem += pulp.lpSum(in_xi[i] for i in available.index) == 11
    for position, (min_count, max_count) in {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}.items():
        idx = available.index[available["position"] == position]
        problem += pulp.lpSum(in_xi[i] for i in idx) >= min_count
        problem += pulp.lpSum(in_xi[i] for i in idx) <= max_count

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"No se encontró una solución óptima (status: {status})")

    squad_idx = [i for i in available.index if in_squad[i].value() == 1]
    starter_idx = [i for i in available.index if in_xi[i].value() == 1]

    squad = available.loc[squad_idx]
    starters = available.loc[starter_idx].sort_values("xp_next", ascending=False).reset_index(drop=True)
    bench = _order_bench(squad, starters)
    return starters, bench


def _print_team(title: str, starters: pd.DataFrame, bench: pd.DataFrame) -> None:
    formation = "-".join(
        str(len(starters[starters["position"] == pos])) for pos in ["DEF", "MID", "FWD"]
    )
    cost = starters["now_cost"].sum() + bench["now_cost"].sum()
    captain, vice_captain = starters.iloc[0], starters.iloc[1]

    cols = ["web_name", "team_name", "position", "now_cost", "xp_next", "next_opponent"]
    print(f"\n=== {title} — Formación {formation} — £{cost:.1f}m / £{BUDGET:.1f}m ===")
    print("\n-- Titulares --")
    print(starters[cols].to_string(index=False))
    print("\n-- Banco (en orden de ingreso) --")
    print(bench[["bench_order"] + cols].to_string(index=False))
    print(f"\nCapitán: {captain['web_name']}  |  Vice-capitán: {vice_captain['web_name']} "
          f"(hereda la cinta si el capitán no juega)")

    print("\n-- Auto-suplencias (quién entra si un titular no juega) --")
    for starter in starters.itertuples():
        sub_name = _first_valid_sub(starter, starters, bench)
        print(f"  {starter.web_name} → {sub_name if sub_name else 'SIN COBERTURA VÁLIDA'}")


def recommend_squad() -> None:
    """Equipo Base autogestionado: se importa evolve_base_squad acá adentro
    (no arriba del archivo) porque transfer_advisor.py ya importa de este
    módulo — un import circular a nivel de módulo rompería la carga. Al
    diferirlo hasta que la función se llama, ambos módulos ya terminaron
    de cargar y el ciclo no es un problema.
    """
    from gameweek_lab.analysis import add_horizon_expected_points
    from gameweek_lab.transfer_advisor import evolve_base_squad

    players = add_horizon_expected_points(add_expected_points(build_players_dataset()))
    starters, bench, log = evolve_base_squad(players)
    print("\n".join(f"  {line}" for line in log))
    _print_team("Equipo Base (autogestionado)", starters, bench)


def recommend_wildcard_squad() -> None:
    players = add_expected_points(build_players_dataset())
    starters, bench = select_wildcard_squad(players)
    _print_team("Equipo Wildcard (banco mínimo, XI máximo)", starters, bench)


def _team_to_rows(starters: pd.DataFrame, bench: pd.DataFrame, squad_type: str) -> pd.DataFrame:
    """Aplana titulares + banco de un equipo a filas 'tidy' (una fila por
    jugador), con columnas categóricas para filtrar/agrupar en Tableau."""
    captain_id = starters.iloc[0]["id"]
    vice_captain_id = starters.iloc[1]["id"]

    starters = starters.copy()
    starters["role"] = "Titular"
    bench = bench.copy()
    bench["role"] = "Banco"

    team = pd.concat([starters, bench], ignore_index=True)
    team["squad_type"] = squad_type
    team["is_captain"] = team["id"] == captain_id
    team["is_vice_captain"] = team["id"] == vice_captain_id
    # bench_order: 0 = titular; 1 = arquero suplente (slot fijo); 2-4 = orden de ingreso
    team["bench_order"] = team["bench_order"].fillna(0).astype(int) if "bench_order" in team else 0
    return team


def export_squads_for_tableau(
    players: pd.DataFrame | None = None,
    base_starters: pd.DataFrame | None = None,
    base_bench: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula ambos equipos (Base y Wildcard) y los guarda en un solo CSV
    tidy — una fuente de datos lista para conectar directo en Tableau,
    con columnas para distinguir equipo, titular/banco y capitanía.

    Acepta un `players` ya calculado (por ejemplo, con las fotos ya
    resueltas por resolve_photo_urls) para no recalcular todo de nuevo;
    si no se pasa nada, lo arma desde cero.

    El Base también se puede pasar ya calculado (`base_starters`/`base_bench`)
    — es lo normal en producción, porque el Base real es el equipo
    autogestionado y persistido (evolve_base_squad en transfer_advisor.py),
    no uno recalculado desde cero cada vez. Sin esos parámetros, cae al
    comportamiento anterior (from-scratch) como referencia/comparación.
    """
    if players is None:
        players = add_expected_points(build_players_dataset())

    if base_starters is None or base_bench is None:
        base_squad = select_squad(players)
        base_starters, base_bench = select_starting_xi(base_squad)
    wildcard_starters, wildcard_bench = select_wildcard_squad(players)

    combined = pd.concat([
        _team_to_rows(base_starters, base_bench, "Base"),
        _team_to_rows(wildcard_starters, wildcard_bench, "Wildcard"),
    ], ignore_index=True)

    columns = [
        "squad_type", "role", "bench_order", "is_captain", "is_vice_captain",
        "web_name", "team_name", "position", "now_cost", "xp_next", "photo_url",
        "next_opponent", "next_is_home", "next_fixture_difficulty",
    ]
    combined = combined[columns]

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "squad_recommendations.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado {out_path} ({len(combined)} filas)")
    return combined


if __name__ == "__main__":
    recommend_squad()
