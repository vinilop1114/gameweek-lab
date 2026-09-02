# The Gameweek Lab — FPL Data Pipeline

Scripts locales en Python para analizar datos de Fantasy Premier League (FPL),
como base de contenido para la cuenta de Instagram "The Gameweek Lab".

## Uso

```bash
pip install -r requirements.txt

python scripts/run_fetch.py           # descarga datos frescos de la API de FPL
python scripts/run_analysis.py        # calcula xP, diferenciales y capitanía
python scripts/run_squad.py           # evoluciona el equipo Base autogestionado (my_team.csv)
python scripts/run_wildcard_squad.py  # equipo Wildcard: banco mínimo, XI máximo, sin memoria
python scripts/export_for_tableau.py  # evoluciona el Base + exporta los tres equipos para Tableau
python scripts/run_trajectory_preview.py  # vista especulativa del equipo a 4 fechas (solo lectura)
python scripts/run_calibration.py     # graba predicción de la fecha + compara xP vs. puntos reales
python scripts/run_transfer_advisor.py --team my_team.csv --bank 0.5 --free-transfers 1 --stance neutral
```

## Estructura

```
gameweek_lab/
  config.py            # rutas y URLs de la API
  fetch.py              # descarga bootstrap-static, fixtures y event/{id}/live → data/raw/*.json
  build_dataset.py      # limpia y combina el JSON crudo → data/processed/players.csv
  analysis.py            # xP (próximo GW y horizonte de 4), diferenciales, capitanía, estrategia de rank
  squad_builder.py       # Wildcard (ILP) + piezas compartidas (XI, banco, auto-suplencias)
  transfer_advisor.py    # asesor manual + evolve_base_squad (motor del Base autogestionado)
  photos.py              # verificación/fallback de fotos de jugadores
  calibration.py         # loop de calibración: xP proyectado vs. puntos reales
data/
  raw/                  # snapshots JSON tal cual los devuelve la API (gitignoreado)
  processed/             # CSVs finales + estado persistido — todo público, versionado
docs/
  chips-strategy.md      # reglas guía para el timing de chips
scripts/                 # un entry point por módulo, ver "Uso" arriba
my_team.csv               # equipo Base persistido (público) — lo edita el propio pipeline
my_team.example.csv       # plantilla de referencia (formato del archivo)
```

## Cómo funciona el cálculo de xP

`xp_next = base_rate × fixture_multiplier × playing_probability`

- **base_rate**: puntos esperados por 90 minutos, calculados desde
  estadísticas subyacentes — `expected_goals_per_90 × puntos_por_gol(posición)`
  `+ expected_assists_per_90 × 3` (ataque), más
  `exp(-expected_goals_conceded_per_90) × puntos_por_clean_sheet(posición)`
  (defensa, con la probabilidad de clean sheet estimada vía Poisson), más
  los puntos esperados por **contribución defensiva** (ver abajo), más
  2 puntos fijos de aparición. **No usa `form`/`points_per_game`** —
  puntos ya anotados mezclan la calidad real del jugador con suerte
  puntual (un bonus de un partido aislado no dice nada sobre si se
  repite); xG/xA reflejan las oportunidades generadas, un proxy más
  estable. Limitación conocida: no modela bonus points (BPS) ni atajadas
  de arquero — FPL no expone un "bono esperado" por jugador.
- **DEFCON (contribución defensiva)**: FPL paga 2 puntos fijos al superar
  un umbral de acciones defensivas (10 CBIT para defensores, 12 CBIRT
  para medios y delanteros; los arqueros no reciben). Como el pago es
  fijo, lo que se estima es la **probabilidad** de superar el umbral, con
  Poisson sobre `defensive_contribution_per_90`. Los defensores promedian
  ~0.33 puntos esperados; los delanteros, casi cero (su umbral de 12 es
  difícil de alcanzar). La tasa propia se mezcla con la mediana de su
  posición, calculada **solo entre quienes jugaron** — los jugadores con
  cero minutos hundían esa referencia de 7.0 a 4.5.
- **fixture_multiplier**: invierte el FDR de FPL (1=fácil, 5=difícil),
  centrado en 1.0 para dificultad media (3). El FDR ya distingue localía
  (verificado: 140 de 200 partidos tienen dificultad distinta para local
  y visitante), y `xp_horizon` suma el multiplicador de **cada partido
  individual** — no promedia dificultad.
- **availability = start_rate × playing_probability** — dos cosas
  distintas que se modelan por separado:
  - **`start_rate`** (rotación): con qué frecuencia el jugador es
    titular, desde el campo `starts` de la API dividido por los partidos
    de su equipo. Se usa `starts` y no `minutes` porque distingue al
    titular fijo del suplente que suma minutos entrando. Lleva suavizado
    bayesiano hacia `START_RATE_PRIOR` (0.75) con peso
    `START_RATE_PRIOR_WEIGHT` (5 partidos) para que "2 de 2" no se lea
    como 100% titular al arrancar la temporada.
    **El reseteo de temporada** (aplica a TODO el modelo, no solo a la
    rotación): FPL pone en cero todos los acumulados al empezar una
    temporada — `starts`, `minutes` y también las tasas por 90' que son
    el insumo principal del xP. Sin protección, en GW2 el xG90 de un
    jugador saldría de un único partido (ruido puro) y un titular
    indiscutido sería indistinguible de un suplente.

    `save_last_season_baseline` congela esas métricas en pre-temporada,
    mientras todavía están disponibles, en
    `data/processed/last_season_baseline.csv`. Después se usan de dos
    formas: como **prior** para `start_rate`, y como **mezcla ponderada**
    para las tasas por 90' (`_blend_with_baseline`) — al principio manda
    la temporada anterior, y los datos nuevos van ganando peso hasta
    dominar al llegar a `BASELINE_BLEND_MINUTES` (900 min ≈ 10 partidos).
    Verificado: tras GW1, sin baseline Thiago y Nmecha darían ambos 0.79
    de titularidad; con baseline dan 0.98 y 0.39.

    **El baseline solo se usa si tiene minutos detrás** (`MIN_BASELINE_MINUTES`).
    Un `start_rate` de 0.00 confundía dos cosas muy distintas: "era
    suplente" y "no jugó en esta liga" — y **200 de 600 jugadores del
    baseline tienen exactamente cero minutos** (equipos ascendidos,
    fichajes, cesiones que volvieron). Rashford, Mendy, Ajayi y Sangaré
    arrancaron GW1/GW2 cargando un prior de 0.00 que no venía de
    suplencia sino de ausencia de datos; Mendy y Ajayi estuvieron entre
    los mejores puntajes de la fecha contra una proyección de 0.35. El
    corte es 1 minuto y no un número mayor: cero es inequívoco, mientras
    que 300 minutos sí dicen algo sobre el rol del jugador.
  - **`playing_probability`** (lesión): `chance_of_playing_next_round` /
    100, o 100% si no hay duda reportada. **Ojo:** verificado contra
    datos reales, solo 9 de 224 jugadores elegibles tienen valor no nulo
    — ese campo solo se llena ante lesión reportada, así que por sí solo
    es un no-op para el 96% del pool. Sin `start_rate`, el modelo trataba
    igual a un titular indiscutido y a un suplente habitual sano.

Jugadores con menos de `MIN_MINUTES_FOR_RANKING` (900 min, ~10 partidos) se
excluyen de diferenciales y capitanía — con poca muestra, el promedio de
puntos es ruido, no señal.

## Techo y probabilidad de haul

`xp_next` responde "cuánto saca en promedio", pero para capitanía eso
engaña: duplicar a un jugador de 5 xP consistente no es lo mismo que
duplicar a uno de 5 xP que alterna entre blanks y hauls.
`add_ceiling_metrics` construye la **distribución completa** de puntos —
goles ~ Poisson(xG90 × dificultad), asistencias ~ Poisson(xA90 ×
dificultad), clean sheet ~ Bernoulli, más el escenario "no juega" que
aporta 0 — y de ahí saca dos métricas:

- **`xp_ceiling`**: percentil 90 de puntos. "En su 10% de mejores
  partidos, saca al menos esto."
- **`haul_probability`**: P(puntos ≥ 10), el umbral clásico de haul en
  FPL. Muy directo de comunicar.

El contraste real que motivó esto, con datos de hoy:

| Jugador | xP medio | Techo | P(haul) |
|---|---|---|---|
| Raya (GKP) | 4.94 | 6.0 | 0.0% |
| Haaland (FWD) | 4.71 | 10.0 | 16.3% |

Raya proyecta **más** xP promedio, pero un arquero no puede hacer un
haul — su distribución es estrecha. El modelo lineal no distinguía esos
dos casos.

**Dónde se usa:** solo en capitanía. `captaincy_picks` siempre muestra
techo y probabilidad de haul, y con `--stance chase` ordena por techo en
vez de por media (remontar posiciones necesita resultados grandes, no
consistencia). Las transferencias siguen usando el promedio a 4 fechas:
ahí la varianza semana a semana se diluye y lo que importa es el
acumulado.

**Efecto secundario del factor de rotación:** modelarlo hizo innecesaria
una restricción dura de "profundidad viable por posición". Un suplente
habitual ahora proyecta poco (Nmecha, 10 titularidades, pasó de 4.83 a
1.54 xP), así que el optimizador lo evita **solo**. El equipo rearmado
quedó con 5/5 MID y 2/3 FWD por encima del 70% de titularidad — el
mismo mínimo que se hubiera impuesto a mano, pero emergente en vez de
cableado.

## Cómo funciona la recomendación de equipo

Armar los 15 titulares de un plantel (`select_squad`) es un problema de
optimización con restricciones cruzadas (presupuesto, exactamente
2 GKP/5 DEF/5 MID/3 FWD, máx. 3 por club, sin enfrentamientos internos —
ver abajo). Se resuelve con programación lineal entera (`PuLP`), que
garantiza el óptimo real — no una aproximación. El Wildcard usa esto
directo (recalculado desde cero cada vez); el Base lo usa solo como punto
de partida — ver la sección siguiente, es la pieza más importante del
proyecto.

**11 titulares** (`select_starting_xi`): de los 15 ya elegidos (Base o
Wildcard), se prueban las ~13 formaciones válidas (3-5 DEF, 2-5 MID,
1-3 FWD) y se toma la de mayor xP. Al ser un espacio de búsqueda chico,
alcanza con fuerza bruta — no hace falta un solver acá.

Solo se consideran jugadores con `status == "a"` (disponibles, sin lesión ni
suspensión) y al menos `MIN_MINUTES_FOR_RANKING` minutos — mismo filtro que
en diferenciales/capitanía, por la misma razón: evitar que un jugador con
muestra chica (ej. un arquero con un solo partido bueno) desplace del
plantel a un titular real por ruido estadístico.

**Enfrentamientos internos — permitidos, pero pagan su precio:** tener al
defensor/arquero de un equipo y a un mediocampista/delantero del rival que
enfrenta ese mismo gameweek tiene un costo real (si el atacante anota, mata
el clean sheet del propio defensor). En vez de prohibir esas duplas, el
modelo les descuenta ese costo esperado (`CLASH_PENALTY_XP` ≈ 0.5 xP,
ver `_internal_clash_penalty` en `squad_builder.py`): una dupla que proyecte
ganar más que su penalización entra al equipo igual. El auto-sabotaje está
permitido cuando los números lo justifican. En el Wildcard la penalización
solo mira a los titulares — un suplente que no juega no le rompe el clean
sheet a nadie.

**Simplificación conocida:** cuando se arma desde cero (Wildcard, o el
Base la primera vez), el modelo maximiza el xP total de los 15, no
específicamente el de los 11 mejores. En la práctica esto tiende a
"gastar de más" en el 5to defensor o 2do arquero en vez de priorizar
jugadores baratos y confiables para el banco.

## Equipo Base vs Equipo Wildcard

Dos filosofías completamente distintas, no solo dos parámetros distintos
del mismo cálculo:

- **Wildcard** (`select_wildcard_squad`): un ejercicio *sin memoria* —
  "si pudiera rearmar todo desde cero hoy, con las reglas de FPL, ¿cuál es
  el mejor equipo posible?". Se recalcula 100% desde cero cada vez, sin
  importar qué recomendó ayer. Como el banco no necesita ser jugable
  (nunca se sostiene más de una fecha), gasta lo mínimo posible ahí y
  vuelca el resto del presupuesto a los 11 titulares. Un solo ILP decide
  plantel y titulares a la vez (variables `in_squad` e `in_xi`, con
  `in_xi <= in_squad`): maximizar el xP del XI es la prioridad absoluta
  (peso `WILDCARD_XI_WEIGHT`), y minimizar el costo total del plantel es
  el criterio de desempate. Los enfrentamientos internos solo se evalúan
  entre titulares — un suplente que no juega no le puede romper el clean
  sheet a nadie.

- **Base** (`evolve_base_squad` en `transfer_advisor.py`): el equipo que
  **de verdad vas a usar en FPL**, con memoria real. No se recalcula desde
  cero — parte del equipo de la fecha anterior (persistido en
  `my_team.csv`) y el modelo decide, con la misma lógica del asesor de
  transferencias (ver abajo), si vale la pena mover algo esta semana. Si
  no hay una mejora que justifique gastar la transferencia, el Base se
  queda exactamente igual. Si la hay, aplica el cambio — hits de -4
  incluidos, si el modelo los considera rentables (ver "Asesor de
  transferencias").

  Tiene **dos frenos**, no uno:

  1. *Una sola vez por gameweek* — un archivo de estado
     (`data/processed/base_squad_state.json`) guarda la última fecha
     evaluada y las transferencias acumuladas, para no "gastar" cambios
     distintos cada día de la misma semana.
  2. *Solo en las últimas `TRANSFER_DECISION_WINDOW_HOURS` (3h) antes del
     deadline* — sin esto, el primer freno dispararía la decisión apenas
     termina la fecha anterior: el lunes, con la peor información de la
     semana. Las lesiones se confirman en las conferencias de jueves y
     viernes, y los precios de FPL se mueven a diario.

  `python scripts/run_squad.py --force` saltea la ventana (no el primer
  freno) para ver qué haría sin esperar al deadline.

- **Base proyectado** (`preview_base_transfers`): el Base con la
  transferencia propuesta **ya aplicada**, calculado todos los días y
  publicado como un tercer `squad_type` en `squad_recommendations.csv`.
  No toca `my_team.csv` ni el archivo de estado — es puramente de
  lectura.

  Existe porque los dos frenos de arriba, que son correctos para
  *decidir*, eran un problema para *publicar*: hasta 3 horas antes del
  deadline el CSV mostraba el equipo viejo, y el cambio recién aparecía
  cuando ya no quedaba tiempo de preparar un post. La decisión definitiva
  sigue tomándose en la ventana del deadline, con las lesiones ya
  confirmadas; el proyectado es la mejor apuesta con los datos de hoy y
  puede cambiar. `plan_transfers()` es el núcleo de decisión compartido:
  `evolve_base_squad` y `preview_base_transfers` lo llaman con los mismos
  argumentos y difieren solo en si persisten el resultado, así que el
  proyectado no puede desviarse de lo que el modelo realmente haría.

En la práctica, el XI del Wildcard suele sacar más xP que el del Base
(mismo presupuesto, banco mucho más barato) — es la comparación esperada:
uno es el techo teórico sin restricciones reales, el otro es el equipo
real, restringido por lo que ya comprás y cuántas transferencias tenés.

**Capitán, vice y orden del banco:** cada equipo recomienda capitán (mayor
xP del XI) y vice-capitán (segundo mayor — hereda la cinta si el capitán no
juega). El banco sale ordenado según la mecánica real de FPL: el arquero
suplente en su slot fijo (los arqueros solo se intercambian entre sí) y los
tres de campo por xP descendente — poner al mejor primero no arriesga nada,
porque cuando un titular no juega FPL recorre el banco en orden y saltea
automáticamente a quien rompería la formación mínima (≥3 DEF, ≥2 MID,
≥1 FWD). La salida incluye una simulación de auto-suplencias: para cada
titular, quién entraría si no juega, ya validado contra esa regla.

## Asesor de transferencias

Dos formas de usar la misma lógica de decisión:

1. **Automática** (`evolve_base_squad`, corre dentro de `run_squad.py` y
   `export_for_tableau.py`): aplica los cambios directo sobre
   `my_team.csv`, sin pedir confirmación. Es lo que mantiene al equipo
   Base al día — ver la sección anterior.
2. **Manual/interactiva** (`python scripts/run_transfer_advisor.py --team
   my_team.csv --bank 0.5 --free-transfers 1 --stance neutral`): la misma
   lógica, pero solo imprime la recomendación para que la leas — no toca
   `my_team.csv`. Útil para simular "¿qué pasaría si tuviera este
   banco/estas transferencias, o si jugara a proteger/remontar rank?" sin
   afectar el estado real del Base.

**`--stance` (protect/neutral/chase):** el xP puro no tiene opinión sobre
riesgo. Proteger una buena posición en tu liga conviene con baja varianza
(jugadores de alto ownership — si fallan, le fallan a todos tus rivales
por igual); remontar desde atrás conviene con alta varianza (diferenciales
de bajo ownership, que si aciertan te separan del resto — un template no
te separa de nadie aunque rinda). `add_rank_adjusted_value` (en
`analysis.py`) calcula `rank_value` = xP ajustado por el **percentil** de
ownership del jugador dentro del propio pool evaluado (no un umbral fijo
tipo "50%" — la distribución real está muy sesgada: mediana ~1.6% entre
candidatos confiables, así que el percentil se autocalibra). Con
`stance="neutral"` (default), `rank_value = xP`, sin cambios de
comportamiento. Afecta tanto el ranking de cambios como la capitanía
sugerida — la capitanía es donde más pesa, por el doble puntaje.
El equipo Base automático (`evolve_base_squad`) se queda siempre en
`neutral` — es un sistema sin humano en el loop, no tiene sentido que se
vuelva más arriesgado sin que se lo pidas explícitamente cada vez.

Ambas responden las mismas preguntas:

- **El mejor cambio disponible** esta fecha, rankeado por ganancia de xP en
  las próximas `HORIZON_GAMEWEEKS` fechas (4 por defecto) — no solo la
  próxima, así el cambio "se prepara" para el calendario que viene.
- **Si conviene hacerlo o guardar la transferencia**: mejoras marginales
  (< `BANK_THRESHOLD` xP en el horizonte) no justifican gastar la
  transferencia — acumularla (hasta 5) habilita movimientos dobles después.
- **Cuántas transferencias libres hay realmente** (`_transfers_granted`):
  el Base automático acredita **1 por fecha, salvo GW1, que no da
  ninguna** — el plantel inicial se arma sin límite de cambios, así que
  la primera transferencia libre recién aparece en GW2. Sin esto el
  modelo arrancaba la temporada creyendo tener una de más y llegó a
  proponer dos cambios teniendo una sola disponible (detectado en la
  práctica, GW2). El saldo acumulado vive en `banked_free_transfers`
  dentro del archivo de estado.
- **Si un hit de -4 se justifica**: solo si el segundo mejor cambio gana más
  de 4 + `HIT_UNCERTAINTY_MARGIN` puntos proyectados — pagar puntos ciertos
  por una proyección requiere margen de seguridad, no empate técnico.
- **Jugadores con bandera** (lesión/suspensión/duda) se marcan siempre, y
  un cambio que saca a uno de ellos se recomienda aunque la ganancia sea chica.
- **La ganancia se mide sobre el XI, no sobre el jugador suelto**
  (`xi_horizon_gain`). En cada posición juega un número fijo de titulares
  por fecha, así que lo que vale un cambio no es `xP(entra) − xP(sale)`
  sino cuánto sube el mejor XI posible de cada fecha, ya contando al
  resto del plantel. La diferencia es enorme en el arco, donde tenés dos
  arqueros y juega uno: si el suplente ya iba a arrancar en tres de las
  cuatro fechas, el arquero que entra se mide contra **ese suplente**, no
  contra el que sale.

  El caso que lo motivó (GW3): el modelo valuó Roefs → Verbruggen en
  +3.86 y gastó la transferencia libre. Contando que Raya ya estaba en el
  banco y habría arrancado en GW4-GW6, la ganancia real era **+1.33** —
  por debajo del umbral, o sea que correspondía guardarla. Sobre el
  plantel real, la valuación vieja sobrestimaba la ganancia en 1.17 xP
  (mediana) y hasta 3.13 en el peor caso, y dejaba 10 cambios por encima
  del umbral donde en realidad había 3.
- **Si el que entra no arranca en ninguna fecha del horizonte**, el XI no
  gana nada, así que la ganancia se cobra al `BENCH_GAIN_DISCOUNT` (0.3)
  del avance bruto. Sigue siendo una penalización suave y no una
  prohibición: el cambio es posible si la ventaja alcanza igual. Si en
  cambio arranca en *algunas* fechas, ya no hace falta descuento alguno —
  el cálculo por fecha lo cobra exactamente en las semanas que juega.
- **El presupuesto usa el precio de venta real de FPL, no el precio de
  mercado**: `my_team.csv` guarda `purchase_price` (precio al que
  compraste cada jugador) además de `web_name`/`team_name`. Si un jugador
  subió de precio desde que lo compraste, FPL solo te devuelve la mitad
  de la ganancia al venderlo (redondeada hacia abajo al escalón de
  £0.1m) — comprado en £5.0m, ahora vale £5.3m, se vende en £5.1m, no en
  £5.3m. Si bajó o quedó igual, vendés al precio actual completo. Cuando
  se aplica un cambio (`_apply_swap`), el jugador que entra queda
  registrado como comprado a su precio de hoy. `advise()` muestra el
  valor de mercado del equipo, lo gastado, y cuánto recuperarías
  vendiendo todo — la "ganancia de valor" que FPL premia a lo largo de
  la temporada, ahora visible.

El xP a horizonte suma los partidos reales de cada equipo por fecha, así que
double gameweeks (dos partidos) y blanks (ninguno) quedan contados
naturalmente. La duda de disponibilidad de un lesionado se asume persistente
en el horizonte — conservador a propósito.

**Lo que el asesor NO evalúa: financiar un puesto vendiendo en otro.** Cada
cambio recibe un presupuesto de `banco + precio de venta de ese mismo
jugador`, y los dos movimientos que permite por fecha se evalúan con banco
en cero, así que la plata que sobra del primero nunca financia al segundo.
"Bajo el arquero suplente para subir un defensor" —una jugada común en
FPL— no aparece en las recomendaciones porque **no existe en su espacio de
búsqueda**, no porque la haya evaluado y descartado.

Medido en GW3: el aporte marginal del segundo arquero, dado que el otro ya
está en el plantel, era de 0.14 xP en 4 fechas (£5.0m comprando casi nada,
porque el otro arranca en 3 de las 4 igual). Liberando esa plata había un
plan de dos movimientos que valía +4.90 xP contra los +2.64 del mejor
movimiento único. Se dejó sin implementar a propósito: buscar pares
multiplica el espacio de búsqueda (~35 candidatos pasan a ~600 pares) y
gasta 2 transferencias en vez de 1 — con una sola libre eso es un hit de
-4, que la ganancia no llega a justificar. El Wildcard captura esta
jugada solo, porque reparte todo el presupuesto desde cero.

**Rendimiento:** `find_best_swaps` evalúa cientos de candidatos por
corrida, y para cada uno necesita saber si terminaría de titular o en el
banco. Hacerlo reconstruyendo el equipo con pandas y llamando a
`select_starting_xi` (el enfoque original) escalaba mal: con el modelo de
xP actual, que genera más candidatos "cercanos" que el anterior, una sola
llamada a `find_best_swaps` tardaba ~14s, y la vista de 4 fechas
(`simulate_squad_trajectory`, que la llama varias veces) casi 80s.
`_best_xi_total` (en `squad_builder.py`) es `select_starting_xi` reducido
a aritmética: listas de Python de como mucho 5 elementos por posición,
sin ninguna operación de pandas. Hoy `xi_horizon_gain` lo llama una vez
por fecha del horizonte y por candidato, y aun así la llamada completa
corre en ~0.2s. Verificado con una prueba diferencial contra el camino
lento de verdad — aplicar el cambio y correr `select_starting_xi` una vez
por fecha — sobre 974 casos reales, 0 discrepancias. A diferencia del
filtro de `raw_gain > 0` (que ayuda pero depende de cuántos candidatos
pasen), esta es una mejora estructural: no se degrada aunque el modelo
cambie y aparezcan más candidatos cercanos.

El timing de chips no se optimiza (requeriría proyectar la temporada
completa): hay reglas guía en `docs/chips-strategy.md`, y el asesor avisa
cuando detecta un double gameweek en el horizonte.

Filosofía: **prepararse, no predecir** — el equipo Base itera fecha a fecha
con datos frescos; no arma un plan de 10 fechas que la realidad va a romper
en la primera. `my_team.csv` es público en el repo a propósito, junto con
todos los demás CSVs — el Project de claude.ai también puede analizarlo.

**Jugadas a balón parado:** `set_piece_duties` marca quién es **primera
opción** en penales, tiros libres directos y córners (B.Fernandes:
"Penales, Tiros libres, Córners"). 43 jugadores tienen alguna función.

Es información de contexto — **no entra al cálculo de xP**, y la razón
importa: `expected_goals` de FPL **ya incluye los penales que el jugador
pateó**. Multiplicar su xG por "es penalty taker" sería contar los
penales dos veces. El ajuste correcto sería sumar solo cuando alguien es
ejecutor *ahora* pero su histórico no lo refleja (llegó al club, o el
ejecutor anterior se fue) — y eso no se puede distinguir: la API no
expone xG sin penales (npxG) ni penales convertidos, verificado tanto en
`bootstrap-static` como en el detalle por jugador. Se expone el dato
para que la decisión la tome una persona con contexto: hay casos
visibles donde importa, como un ejecutor de penales con xG histórico
casi nulo por haber cambiado de liga.

**Momentum de transferencias:** `players_scored.csv` incluye
`transfers_in_event`/`transfers_out_event` (transferencias netas de la
comunidad esta fecha, señal de posibles subidas/bajadas de precio antes
de que pasen). Están en 0 para todos en pre-temporada — armar tu plantel
inicial no cuenta como "transferencia" en FPL, así que no hay señal real
todavía. Quedan expuestos para cuando arranque la temporada; no se
integraron a ninguna decisión automática porque anticipar el algoritmo
de precios de FPL con precisión (no está publicado oficialmente) sería
una predicción, no una preparación — justo lo que este proyecto evita.

## Loop de calibración

`python scripts/run_calibration.py` — la única forma de saber si el xP le
está pegando bien es comparar, fecha a fecha, lo que proyectó contra lo
que realmente pasó. Sin esto, no hay manera de detectar si el modelo
sobrestima delanteros o subestima defensas baratas — solo la sensación de
que "parece razonable".

Dos pasos, gateados para no duplicar trabajo si el pipeline corre varias
veces el mismo día:

1. **`snapshot_predictions`**: antes de que se juegue cada fecha, graba
   `xp_next` de cada jugador disponible en `data/processed/xp_calibration.csv`
   (append, no se sobreescribe). Sin esto la proyección se pierde apenas
   se refrescan los datos al día siguiente — `players_scored.csv` no
   guarda historial, siempre muestra el estado actual.
2. **`record_actual_points`**: una vez que **todos los partidos de una
   fecha terminaron**, completa los puntos reales usando
   `/event/{id}/live/` — a diferencia de `event_points` en
   bootstrap-static (que solo refleja la fecha "actual" del juego y se
   pisa apenas arranca la siguiente), este endpoint da los puntos de esa
   fecha puntual sin ambigüedad, así que no importa si el pipeline se
   atrasa unos días. El corte mira `finished_provisional` en los
   fixtures, **no `data_checked` en los events**: verificado en GW1, los
   10 partidos habían terminado con los bonus ya asignados mientras
   `data_checked` seguía en `False` — FPL lo marca tras su verificación
   final, que puede tardar días, y esperarlo dejaba la calibración
   parada sobre datos ya completos.

`build_calibration_report` hace tres cortes:

- **Sesgo separado por población** (disponibilidad vs. scoring). El
  agregado promedia dos grupos que fallan en direcciones opuestas y se
  cancelan: `-0.76` para quienes no aparecieron, `+1.63` para quienes
  jugaron. Un solo número no permite saber cuál de los dos problemas
  está mejorando. El reporte además descuenta el **artefacto de
  condicionar**: a un jugador con 60% de probabilidad de arrancar se le
  acreditan 1.2 de los 2 puntos de aparición, así que mirando solo a
  quienes jugaron el modelo queda corto **por diseño** — lo que importa
  es el sesgo que queda por encima de eso.
- **Poder de ordenamiento** contra baselines (`ep_next` de FPL,
  `points_per_game`, `selected_by_percent`, `now_cost`). El sesgo es una
  constante que se puede sumar después; lo que el modelo realmente vende
  es el ranking. Se mide con Spearman, calculado como Pearson sobre
  rangos para no agregar scipy por una transformación de una línea.
  Primera lectura: `selected_by_percent` 0.430 vs `xp_next` 0.423 — si
  un predictor trivial le gana de forma sostenida, la complejidad no se
  está pagando.
- **Sesgo por posición**, solo entre quienes tuvieron minutos.

Con menos de `MIN_GAMEWEEKS_FOR_BIAS_REPORT` (4) **fechas**, el reporte
lo advierte explícitamente. El umbral se cuenta en fechas y no en filas
a propósito: una fecha aporta ~480 observaciones, pero salen de los
mismos 10 partidos y no son evidencia independiente.

**Deliberadamente no auto-corrige nada.** El reporte es para que una
persona lo lea; ajustar `GOAL_POINTS`, `CLEAN_SHEET_POINTS` o cualquier
peso del xP contra dos o tres fechas sería ajustar contra ruido.

## Vista especulativa a 4 fechas

`python scripts/run_trajectory_preview.py` genera
`data/processed/squad_trajectory_preview.csv`: 15 filas (un "slot" del
plantel actual) × columnas `GW1`...`GW4` con el nombre del jugador en ese
slot cada fecha. Si un slot no cambia, el nombre se repite en las columnas
siguientes; si cambia, se ve el nombre nuevo desde esa columna en
adelante — fácil de leer de un vistazo.

**Cada columna es el plantel que juega esa fecha**, o sea con la decisión
de esa fecha ya aplicada — la primera columna incluida. Eso hace que la
columna de hoy sea exactamente el `squad_type = "Base proyectado"` de
`squad_recommendations.csv`; si no coinciden, hay un bug. (Antes el loop
arrancaba en la segunda fecha y sembraba la primera columna con
`my_team.csv` tal cual, así que la transferencia de esta fecha aparecía
recién en la columna siguiente: la vista contradecía al Base proyectado
con una fecha de desfase.) La excepción es cuando la fecha en curso ya se
ejecutó: ahí `my_team.csv` ya tiene sus transferencias, la primera
columna lo dice, y la primera decisión simulada es la de la fecha
siguiente.

**Importante — esto NO es una predicción**, es una simulación: corre
`plan_transfers` —el mismo núcleo de decisión que ejecuta el equipo real,
no una copia— cuatro veces seguidas usando el `xp_horizon` de **hoy**, sin
esperar datos nuevos entre fecha y fecha simulada. En la vida real, cada gameweek trae datos
frescos (lesiones, precios, forma) que probablemente cambien la decisión
real cuando de verdad llegue esa fecha — por eso es "el plan de hoy, si
nada cambiara", no un compromiso. Es de solo lectura: nunca toca
`my_team.csv` ni `base_squad_state.json`, son completamente independientes.

## Briefing para redactar posts

`data/processed/briefing.md` — un resumen en Markdown que genera el mismo
pipeline, pensado para que un modelo redacte posts sin leer los CSVs
completos. Trae equipo Base, equipo Wildcard, top capitanías, top
diferenciales, las transferencias, el repaso de la última fecha cerrada y
una sección que explica cómo interpretar cada métrica.

Dos de esas secciones son material que no existe en ningún CSV:

- **Transferencias** (`_transfer_section`): la propuesta de esta fecha
  (todavía sin aplicar, marcada como tal) y los movimientos ya
  ejecutados, con el motivo y la ganancia de xP de cada uno. Ni
  `my_team.csv` ni `squad_recommendations.csv` distinguen a un jugador
  que entró esta semana de uno que lleva meses — solo muestran el plantel
  resultante, así que sin esta sección la transferencia solo existía en
  la salida de terminal de `run_squad.py`, que se pierde.
- **Repaso de la última fecha** (`_last_gameweek_review`): sesgo del
  modelo, error absoluto medio, sesgo por posición y los 5 mejores
  puntajes reales contra lo que el modelo había previsto — sale del
  historial de calibración. Es material propio para contenido (nadie
  publica el error de su propio modelo) y obliga a que los posts de
  repaso salgan de datos medidos y no de impresiones. Incluye la
  advertencia de que subestimar es el sesgo *esperado*, porque el modelo
  no cuenta bonus points.

Existe por tres razones concretas:

1. **Costo**: `players_scored.csv` son 592 jugadores × 32 columnas (~39k
   tokens) para armar un post que usa ~20 jugadores. El briefing pesa
   ~6 KB — unas 10 veces menos.
2. **Seguridad de los datos**: viene ya filtrado por
   `MIN_MINUTES_FOR_RANKING`, así que no expone la trampa de las tasas
   por 90' infladas (hay jugadores con 2 minutos jugados y el "mejor
   xG90 de la liga").
3. **Menos ambigüedad**: incluye la explicación de cada métrica en el
   propio archivo, en vez de depender de que quien lo lea recuerde qué
   significa `xp_ceiling` o por qué `set_piece_duties` no está sumado al
   xP.

Se arma reusando el DataFrame que ya produjo `export_squads_for_tableau`,
no recalculando los equipos — así el briefing no puede contradecir al
CSV.

## Exports para Tableau

`python scripts/export_for_tableau.py` genera dos CSVs en `data/processed/`:

- **`players_scored.csv`**: los ~590 jugadores con xP, precio, ownership,
  próximo rival, etc. — para dashboards sobre todo el pool.
- **`squad_recommendations.csv`**: los **tres** equipos combinados en
  formato tidy (una fila por jugador, columnas
  `squad_type`/`role`/`is_captain` para filtrar) — una sola fuente de
  datos para compararlos. `squad_type` toma tres valores: `Base` (el
  equipo tal como está hoy en FPL), `Base proyectado` (el mismo, con la
  transferencia propuesta ya aplicada) y `Wildcard` (el techo teórico).
  Cuando no hay transferencia propuesta, `Base` y `Base proyectado` son
  idénticos; la diferencia entre ambos conjuntos *es* el cambio.

Ambos incluyen `xp_horizon` y `fixtures_horizon` (el xP y el calendario
resumido de las próximas 4 fechas, no solo la siguiente — pensado para que
se pueda explicar el *por qué* de una elección, no solo el qué: por
ejemplo, por qué el asesor prefirió a un jugador sobre otro que tenía mejor
xP inmediato pero un calendario peor a mediano plazo).

Ambos incluyen `photo_url` (foto del jugador desde el CDN oficial de
Premier League, armada con el campo `code` de la API — ver
`PLAYER_PHOTO_URL_TEMPLATE` en `config.py`). Para que Tableau la muestre
como imagen: clic derecho sobre `photo_url` en el panel de datos →
Default Properties → Image Role.

**Tamaño de imagen:** usamos 40x40 (no 110x140 ni 250x250) porque Tableau
Public rechaza imágenes de más de 128KB, y los tamaños más grandes lo
superan en algunos casos (depende de cuánto detalle tenga la foto
puntual, no solo de la resolución). 40x40 tiene margen de sobra siempre.

**Jugadores sin foto:** ~35% de los jugadores no tienen foto todavía en
el CDN de la Premier League — sobre todo fichajes grandes de este
mercado (Wirtz, Donnarumma, Cherki, Zubimendi...) sin sesión oficial
todavía. `export_for_tableau.py` chequea cada URL contra la API en
paralelo (`gameweek_lab/photos.py`, `resolve_photo_urls`) y reemplaza
las que fallan por la silueta genérica oficial de FPL
(`Photo-Missing.png`, la misma que usa fantasy.premierleague.com) — así
ninguna celda queda con el ícono roto. Este chequeo solo corre en el
export para Tableau (~7s con ~590 jugadores, usando una sesión HTTP con
pool de conexiones); el resto del pipeline no depende de red extra.
