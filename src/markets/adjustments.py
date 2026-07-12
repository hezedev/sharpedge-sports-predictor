from __future__ import annotations

from typing import Optional

import pandas as pd
from config import settings

from src.markets.true_probability import PredictionFactor


_SOCCER_ADJUST_CFG = (((((settings or {}).get("betting") or {}).get("adjustments") or {}).get("soccer")) or {})


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _first_present(snapshot: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in snapshot.index:
            return _as_float(snapshot.get(column))
    return 0.0


def _selected_value(direction: float, home_value: float, away_value: float, default: float = 0.0) -> tuple[float, float]:
    if direction == 1.0:
        return home_value, away_value
    if direction == -1.0:
        return away_value, home_value
    return default, default


def build_context_adjustments(
    sport: str,
    selection: str,
    snapshot: Optional[pd.Series],
    context: Optional[dict],
) -> list[PredictionFactor]:
    sport = (sport or "").lower()
    selection = (selection or "").lower()
    snapshot = snapshot if snapshot is not None else pd.Series(dtype=float)
    context = context or {}

    home_like = selection in {"home", "home_or_draw", "over", "player1"}
    away_like = selection in {"away", "away_or_draw", "under", "player2"}
    direction = -1.0 if away_like else 1.0
    if selection == "draw":
        direction = 0.0

    adjustments: list[PredictionFactor] = []

    rest_adv = int(context.get("rest_advantage", 0) or 0)
    if rest_adv and direction != 0.0:
        shift = 0.015 if rest_adv == direction else -0.015
        adjustments.append(
            PredictionFactor(
                name="rest_advantage",
                category="schedule",
                value=shift,
                summary="Rest advantage adjustment from schedule context.",
            )
        )

    if context.get("is_playoff"):
        adjustments.append(
            PredictionFactor(
                name="playoff_context",
                category="motivation",
                value=0.0,
                summary="Playoff / high-leverage match context detected.",
            )
        )

    if context.get("playoff_motivation"):
        adjustments.append(
            PredictionFactor(
                name="playoff_motivation",
                category="motivation",
                value=0.0,
                summary="Playoff-race or elimination leverage raises motivation and variance.",
            )
        )

    if context.get("final_day_volatility"):
        adjustments.append(
            PredictionFactor(
                name="final_day_volatility",
                category="motivation",
                value=0.0,
                summary="End-of-season / final-round context increases scoreboard and game-state volatility.",
            )
        )

    home_rest_days = _as_float(context.get("home_rest_days", snapshot.get("home_rest_days")), 0.0)
    away_rest_days = _as_float(context.get("away_rest_days", snapshot.get("away_rest_days")), 0.0)
    selected_rest_days, opponent_rest_days = _selected_value(direction, home_rest_days, away_rest_days)
    if direction != 0.0 and context.get("fixture_congestion_risk"):
        congestion_edge = 0.0
        if selected_rest_days and opponent_rest_days:
            major_congestion_penalty = float(_SOCCER_ADJUST_CFG.get("congestion_major_penalty", 0.014) or 0.014)
            minor_congestion_penalty = float(_SOCCER_ADJUST_CFG.get("congestion_minor_penalty", 0.008) or 0.008)
            if selected_rest_days <= 2.5 and opponent_rest_days >= 4.5:
                congestion_edge = -major_congestion_penalty
            elif opponent_rest_days <= 2.5 and selected_rest_days >= 4.5:
                congestion_edge = major_congestion_penalty
            elif selected_rest_days <= 3.5 and opponent_rest_days >= selected_rest_days + 1.5:
                congestion_edge = -minor_congestion_penalty
            elif opponent_rest_days <= 3.5 and selected_rest_days >= opponent_rest_days + 1.5:
                congestion_edge = minor_congestion_penalty
        if congestion_edge:
            adjustments.append(
                PredictionFactor(
                    name="fixture_congestion",
                    category="schedule",
                    value=round(congestion_edge, 3),
                    summary="Compressed fixture turnaround changes recovery, legs, and squad-management reliability.",
                )
            )
        if selected_rest_days and opponent_rest_days and selected_rest_days <= 3.0 and opponent_rest_days <= 3.0:
            adjustments.append(
                PredictionFactor(
                    name="congestion_volatility",
                    category="schedule",
                    value=0.0,
                    summary="Both sides are on a compressed turnaround, which raises late-breaking variance.",
                )
            )

    away_travel_bucket = int(_as_float(snapshot.get("away_travel_bucket"), 0))
    away_cross_country = int(_as_float(snapshot.get("away_cross_country"), 0))
    away_crossed_2tz = int(_as_float(snapshot.get("away_crossed_2tz"), 0))
    away_tz_shift = _as_float(snapshot.get("away_travel_tz_shift"), 0.0)
    travel_edge = 0.0
    if away_travel_bucket >= 2:
        travel_edge += 0.004 * min(3, away_travel_bucket)
    if away_cross_country:
        travel_edge += 0.004
    if away_crossed_2tz:
        travel_edge += 0.006
    if away_tz_shift >= 2:
        travel_edge += min(0.01, away_tz_shift * 0.0025)
    if direction != 0.0 and travel_edge:
        adjustments.append(
            PredictionFactor(
                name="travel_fatigue",
                category="environment",
                value=round(travel_edge * direction, 3),
                summary="Travel burden adjustment from distance and timezone shift on the away side.",
            )
        )

    weather_risk = int(_as_float(context.get("weather_risk"), 0))
    wind_mph = _as_float(context.get("wind_mph"), 0.0)
    temperature_f = _as_float(context.get("temperature_f"), 0.0)
    precip_mm = _as_float(context.get("precip_mm"), 0.0)
    if weather_risk and direction != 0.0:
        weather_edge = 0.004 * direction
        if wind_mph >= 16:
            weather_edge += 0.002 * direction
        adjustments.append(
            PredictionFactor(
                name="weather_environment",
                category="environment",
                value=round(weather_edge, 3),
                summary="Outdoor weather context leans slightly toward the home side in volatile conditions.",
            )
        )
        adjustments.append(
            PredictionFactor(
                name="weather_uncertainty",
                category="environment",
                value=0.0,
                summary=(
                    f"Weather risk detected (wind {wind_mph:.0f} mph, temp {temperature_f:.0f}F, "
                    f"precip {precip_mm:.1f} mm)."
                ),
            )
        )

    if sport == "soccer":
        dc_xg_diff = _as_float(snapshot.get("dc_xg_diff"))
        xg_diff = _as_float(snapshot.get("xg_diff"))
        home_attack = _as_float(snapshot.get("home_goals_scored_avg"))
        away_attack = _as_float(snapshot.get("away_goals_scored_avg"))
        home_defense = _as_float(snapshot.get("home_goals_conceded_avg"))
        away_defense = _as_float(snapshot.get("away_goals_conceded_avg"))
        h2h_win_rate = _as_float(snapshot.get("h2h_home_win_rate"), 0.5)
        h2h_count = int(_as_float(snapshot.get("h2h_matches_count"), 0))
        home_pts_rate = _as_float(snapshot.get("home_season_pts_rate"), 1.3)
        away_pts_rate = _as_float(snapshot.get("away_season_pts_rate"), 1.3)
        pts_rate_gap = _as_float(snapshot.get("home_pts_rate_vs_away"))

        tactical_edge = max(-0.015, min(0.015, (dc_xg_diff * 0.012) + (xg_diff * 0.008)))
        if direction != 0.0 and tactical_edge:
            adjustments.append(
                PredictionFactor(
                    name="tactical_matchup",
                    category="matchup",
                    value=round(tactical_edge * direction, 3),
                    summary="Soccer matchup edge from expected-goals profile and chance-quality shape.",
                )
            )
        style_clash = ((home_attack - away_defense) - (away_attack - home_defense)) * 0.006
        style_clash = max(-0.012, min(0.012, style_clash))
        if direction != 0.0 and style_clash:
            adjustments.append(
                PredictionFactor(
                    name="style_clash",
                    category="matchup",
                    value=round(style_clash * direction, 3),
                    summary="Attack-vs-defense clash adjustment from rolling scoring and concession profiles.",
                )
            )
        if h2h_count >= 3:
            h2h_edge = max(-0.008, min(0.008, (h2h_win_rate - 0.5) * 0.02))
            if direction != 0.0 and h2h_edge:
                adjustments.append(
                    PredictionFactor(
                        name="h2h_tactical_history",
                        category="coaching",
                        value=round(h2h_edge * direction, 3),
                        summary="Head-to-head pattern proxy for how these setups have historically interacted.",
                    )
                )
        if (home_pts_rate <= 1.05 or away_pts_rate <= 1.05 or abs(pts_rate_gap) <= 0.18):
            adjustments.append(
                PredictionFactor(
                    name="table_pressure",
                    category="motivation",
                    value=0.0,
                    summary="League-table pressure spot detected from season points rates or compressed table gap.",
                )
            )

        selected_pts_rate, opponent_pts_rate = _selected_value(direction, home_pts_rate, away_pts_rate, 1.3)
        if selected_pts_rate and opponent_pts_rate:
            relegation_motivation_boost = float(_SOCCER_ADJUST_CFG.get("relegation_motivation_boost", 0.006) or 0.006)
            opponent_relegation_penalty = float(_SOCCER_ADJUST_CFG.get("opponent_relegation_penalty", -0.006) or -0.006)
            title_motivation_boost = float(_SOCCER_ADJUST_CFG.get("title_motivation_boost", 0.005) or 0.005)
            opponent_title_penalty = float(_SOCCER_ADJUST_CFG.get("opponent_title_penalty", -0.005) or -0.005)
            nothing_to_play_for_penalty = float(_SOCCER_ADJUST_CFG.get("nothing_to_play_for_penalty", -0.006) or -0.006)
            if selected_pts_rate <= 1.05 and opponent_pts_rate >= 1.15:
                adjustments.append(
                    PredictionFactor(
                        name="relegation_motivation",
                        category="motivation",
                        value=relegation_motivation_boost,
                        summary="Selected side profiles like a relegation-pressure team that should bring urgency.",
                    )
                )
            elif opponent_pts_rate <= 1.05 and selected_pts_rate >= 1.15:
                adjustments.append(
                    PredictionFactor(
                        name="opponent_relegation_motivation",
                        category="motivation",
                        value=opponent_relegation_penalty,
                        summary="Opponent profiles like a relegation-pressure side, reducing comfort for the selected team.",
                    )
                )

            if selected_pts_rate >= 1.95 and opponent_pts_rate <= 1.8:
                adjustments.append(
                    PredictionFactor(
                        name="title_motivation",
                        category="motivation",
                        value=title_motivation_boost,
                        summary="Selected side profiles like a title-race team with strong incentive to keep pushing.",
                    )
                )
            elif opponent_pts_rate >= 1.95 and selected_pts_rate <= 1.8:
                adjustments.append(
                    PredictionFactor(
                        name="opponent_title_motivation",
                        category="motivation",
                        value=opponent_title_penalty,
                        summary="Opponent profiles like a title-race team, which raises execution and urgency risk.",
                    )
                )

            if 1.15 <= selected_pts_rate <= 1.55 and (
                opponent_pts_rate <= 1.05 or opponent_pts_rate >= 1.95
            ):
                adjustments.append(
                    PredictionFactor(
                        name="nothing_to_play_for",
                        category="motivation",
                        value=nothing_to_play_for_penalty,
                        summary="Selected side looks more mid-table while the opponent has clearer table incentive.",
                    )
                )

        home_injuries = int(_as_float(context.get("home_injuries_count"), 0))
        away_injuries = int(_as_float(context.get("away_injuries_count"), 0))
        home_susp = int(_as_float(context.get("home_suspensions_count"), 0))
        away_susp = int(_as_float(context.get("away_suspensions_count"), 0))
        home_lineup_confirmed = int(_as_float(context.get("home_lineup_confirmed"), 0))
        away_lineup_confirmed = int(_as_float(context.get("away_lineup_confirmed"), 0))
        home_starters = int(_as_float(context.get("home_likely_starters_count"), 0))
        away_starters = int(_as_float(context.get("away_likely_starters_count"), 0))
        home_spine = int(_as_float(context.get("home_lineup_spine_count"), 0))
        away_spine = int(_as_float(context.get("away_lineup_spine_count"), 0))
        home_goalkeeper_named = int(_as_float(context.get("home_lineup_goalkeeper_named"), 0))
        away_goalkeeper_named = int(_as_float(context.get("away_lineup_goalkeeper_named"), 0))
        home_absences = _as_float(context.get("home_absence_severity"), home_injuries + (home_susp * 2.0))
        away_absences = _as_float(context.get("away_absence_severity"), away_injuries + (away_susp * 2.0))
        home_priority = int(_as_float(context.get("home_priority_absences_count"), home_susp))
        away_priority = int(_as_float(context.get("away_priority_absences_count"), away_susp))
        home_spine_absences = int(_as_float(context.get("home_spine_absences_count"), 0))
        away_spine_absences = int(_as_float(context.get("away_spine_absences_count"), 0))
        absence_edge = max(-0.024, min(0.024, (away_absences - home_absences) * 0.005))
        priority_edge = max(-0.01, min(0.01, (away_priority - home_priority) * 0.004))
        spine_edge = max(-0.01, min(0.01, (away_spine_absences - home_spine_absences) * 0.004))
        total_edge = absence_edge + priority_edge + spine_edge
        if direction != 0.0 and total_edge:
            adjustments.append(
                PredictionFactor(
                    name="availability_edge",
                    category="lineup",
                    value=round(total_edge * direction, 3),
                    summary="Weighted absence gap from live injuries, suspensions, and priority absences.",
                )
            )
        heavy_absence_count = home_absences if direction == 1.0 else away_absences if direction == -1.0 else max(home_absences, away_absences)
        heavy_priority_count = home_priority if direction == 1.0 else away_priority if direction == -1.0 else max(home_priority, away_priority)
        if heavy_absence_count >= 4.0 or heavy_priority_count >= 2:
            adjustments.append(
                PredictionFactor(
                    name="availability_uncertainty",
                    category="lineup",
                    value=0.0,
                    summary="Multiple or high-priority absences on one side may materially change the expected setup.",
                )
            )
        if direction == 1.0 and home_lineup_confirmed and not away_lineup_confirmed:
            adjustments.append(
                PredictionFactor(
                    name="lineup_confirmation",
                    category="lineup",
                    value=0.006,
                    summary="Selected side has a published starting XI while the opponent does not.",
                )
            )
        elif direction == -1.0 and away_lineup_confirmed and not home_lineup_confirmed:
            adjustments.append(
                PredictionFactor(
                    name="lineup_confirmation",
                    category="lineup",
                    value=0.006,
                    summary="Selected side has a published starting XI while the opponent does not.",
                )
            )
        selected_lineup_confirmed = home_lineup_confirmed if direction == 1.0 else away_lineup_confirmed if direction == -1.0 else max(home_lineup_confirmed, away_lineup_confirmed)
        selected_starters = home_starters if direction == 1.0 else away_starters if direction == -1.0 else max(home_starters, away_starters)
        selected_spine = home_spine if direction == 1.0 else away_spine if direction == -1.0 else max(home_spine, away_spine)
        selected_goalkeeper_named = home_goalkeeper_named if direction == 1.0 else away_goalkeeper_named if direction == -1.0 else max(home_goalkeeper_named, away_goalkeeper_named)
        if selected_starters and selected_starters < 11 and not selected_lineup_confirmed:
            adjustments.append(
                PredictionFactor(
                    name="lineup_uncertainty",
                    category="lineup",
                    value=0.0,
                    summary="Expected soccer starting XI is still incomplete or only partially posted.",
                )
            )
        if selected_lineup_confirmed and (selected_spine < 4 or not selected_goalkeeper_named):
            adjustments.append(
                PredictionFactor(
                    name="lineup_shape_uncertainty",
                    category="lineup",
                    value=0.0,
                    summary="Posted starting XI still looks incomplete in the central spine or goalkeeper slot.",
                )
            )
        if context.get("cup_rotation_risk") or context.get("european_rotation_risk"):
            rotation_uncertainty_penalty = float(_SOCCER_ADJUST_CFG.get("rotation_uncertainty_penalty", -0.008) or -0.008)
            adjustments.append(
                PredictionFactor(
                    name="rotation_risk",
                    category="lineup",
                    value=0.0,
                    summary="Cup / continental context raises rotation and squad-management uncertainty.",
                )
            )
            if not selected_lineup_confirmed:
                adjustments.append(
                    PredictionFactor(
                        name="rotation_uncertainty",
                        category="lineup",
                        value=rotation_uncertainty_penalty,
                        summary="Selected side is still unconfirmed in a cup / continental context where rotation risk matters more.",
                    )
                )
        if context.get("final_day_volatility") and not selected_lineup_confirmed:
            late_lineup_risk_penalty = float(_SOCCER_ADJUST_CFG.get("late_lineup_risk_penalty", -0.006) or -0.006)
            adjustments.append(
                PredictionFactor(
                    name="late_lineup_risk",
                    category="lineup",
                    value=late_lineup_risk_penalty,
                    summary="End-of-season context with no confirmed XI raises rotation and late-news risk.",
                )
            )

    if sport == "mlb":
        era_diff = _as_float(snapshot.get("sp_era_diff"))
        whip_diff = _as_float(snapshot.get("sp_whip_diff"))
        k9_diff = _as_float(snapshot.get("sp_k9_diff"))
        bb9_diff = _as_float(snapshot.get("sp_bb9_diff"))
        home_home_wpct = _as_float(snapshot.get("home_home_wpct_20"), 0.5)
        away_away_wpct = _as_float(snapshot.get("away_away_wpct_20"), 0.5)
        home_streak = _as_float(snapshot.get("home_streak"))
        away_streak = _as_float(snapshot.get("away_streak"))
        home_run_form = _as_float(snapshot.get("home_run_diff_10"))
        away_run_form = _as_float(snapshot.get("away_run_diff_10"))
        home_unknown = int(_as_float(snapshot.get("home_sp_unknown"), 0))
        away_unknown = int(_as_float(snapshot.get("away_sp_unknown"), 0))
        home_b2b = int(_as_float(snapshot.get("home_b2b"), 0))
        away_b2b = int(_as_float(snapshot.get("away_b2b"), 0))
        home_games_l3d = _as_float(context.get("home_games_L3D", snapshot.get("home_games_L3D")), 0.0)
        away_games_l3d = _as_float(context.get("away_games_L3D", snapshot.get("away_games_L3D")), 0.0)
        park_factor = _as_float(context.get("park_factor_proxy"), 0.0)

        starter_edge = max(-0.02, min(0.02, (-era_diff * 0.012) + (-whip_diff * 0.025)))
        if direction != 0.0 and starter_edge:
            adjustments.append(
                PredictionFactor(
                    name="starter_quality",
                    category="lineup",
                    value=round(starter_edge * direction, 3),
                    summary="Starting-pitcher quality adjustment from ERA/WHIP gap.",
                )
            )

        if direction == 1.0 and home_unknown and not away_unknown:
            adjustments.append(
                PredictionFactor(
                    name="starter_uncertainty",
                    category="lineup",
                    value=-0.018,
                    summary="Home starter profile is still uncertain / lightly established.",
                )
            )
        elif direction == -1.0 and away_unknown and not home_unknown:
            adjustments.append(
                PredictionFactor(
                    name="starter_uncertainty",
                    category="lineup",
                    value=-0.018,
                    summary="Away starter profile is still uncertain / lightly established.",
                )
            )
        home_confirmed = context.get("home_starter_confirmed")
        away_confirmed = context.get("away_starter_confirmed")
        if direction == 1.0 and home_confirmed is not None and away_confirmed is not None:
            if int(home_confirmed) and not int(away_confirmed):
                adjustments.append(
                    PredictionFactor(
                        name="starter_confirmation",
                        category="lineup",
                        value=0.01,
                        summary="Selected side has a listed probable starter while the opponent still does not.",
                    )
                )
            elif not int(home_confirmed) and int(away_confirmed):
                adjustments.append(
                    PredictionFactor(
                        name="starter_uncertainty",
                        category="lineup",
                        value=-0.01,
                        summary="Opponent has a listed probable starter while the selected side still does not.",
                    )
                )
        elif direction == -1.0 and home_confirmed is not None and away_confirmed is not None:
            if int(away_confirmed) and not int(home_confirmed):
                adjustments.append(
                    PredictionFactor(
                        name="starter_confirmation",
                        category="lineup",
                        value=0.01,
                        summary="Selected side has a listed probable starter while the opponent still does not.",
                    )
                )
            elif not int(away_confirmed) and int(home_confirmed):
                adjustments.append(
                    PredictionFactor(
                        name="starter_uncertainty",
                        category="lineup",
                        value=-0.01,
                        summary="Opponent has a listed probable starter while the selected side still does not.",
                )
            )

        pitch_mix_edge = max(-0.012, min(0.012, (k9_diff * 0.0025) + ((-bb9_diff) * 0.0035)))
        if direction != 0.0 and pitch_mix_edge:
            adjustments.append(
                PredictionFactor(
                    name="pitcher_command",
                    category="matchup",
                    value=round(pitch_mix_edge * direction, 3),
                    summary="Pitcher command and bat-missing edge from K/9 and BB/9 differential.",
                )
            )
        bullpen_edge = max(-0.018, min(0.018, (away_games_l3d - home_games_l3d) * 0.006))
        if direction != 0.0 and abs(bullpen_edge) >= 0.003:
            adjustments.append(
                PredictionFactor(
                    name="bullpen_workload",
                    category="schedule",
                    value=round(bullpen_edge * direction, 3),
                    summary="Bullpen workload proxy from each team's games played over the last three days.",
                )
            )
        if context.get("bullpen_fatigue_risk"):
            adjustments.append(
                PredictionFactor(
                    name="bullpen_fatigue_risk",
                    category="schedule",
                    value=0.0,
                    summary="One or both bullpens are in a compressed recent workload spot.",
                )
            )
        matchup_edge = max(
            -0.012,
            min(0.012, ((home_run_form - away_run_form) * 0.0025) + ((-era_diff) * 0.004)),
        )
        if direction != 0.0 and matchup_edge:
            adjustments.append(
                PredictionFactor(
                    name="starter_form_synergy",
                    category="matchup",
                    value=round(matchup_edge * direction, 3),
                    summary="Starter-vs-form synergy adjustment from recent run differential and probable pitcher quality.",
                )
            )
        if park_factor:
            park_edge = max(-0.006, min(0.006, (park_factor - 1.0) * (home_run_form - away_run_form) * 0.004))
            adjustments.append(
                PredictionFactor(
                    name="park_factor",
                    category="environment",
                    value=round(park_edge * direction, 3) if direction != 0.0 else 0.0,
                    summary=(
                        f"Park run-environment proxy checked ({context.get('park_run_environment', 'neutral')}, "
                        f"factor {park_factor:.3f})."
                    ),
                )
            )
        venue_edge = max(-0.01, min(0.01, (home_home_wpct - away_away_wpct) * 0.03))
        if direction != 0.0 and venue_edge:
            adjustments.append(
                PredictionFactor(
                    name="venue_split_edge",
                    category="coaching",
                    value=round(venue_edge * direction, 3),
                    summary="Home/away split edge proxy for park comfort, tactical fit, and staff routines.",
                )
            )
        if abs(home_streak) >= 4 or abs(away_streak) >= 4:
            momentum_edge = max(-0.006, min(0.006, (home_streak - away_streak) * 0.0012))
            if direction != 0.0 and momentum_edge:
                adjustments.append(
                    PredictionFactor(
                        name="streak_pressure",
                        category="motivation",
                        value=round(momentum_edge * direction, 3),
                        summary="Extended streak context can change urgency, bullpen choices, and late-game management.",
                    )
                )

        if direction == 1.0:
            if home_b2b and not away_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=-0.012,
                        summary="Home side is on a tighter turnaround than the opponent.",
                    )
                )
            elif away_b2b and not home_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=0.012,
                        summary="Away side is on a tighter turnaround than the opponent.",
                    )
                )
        elif direction == -1.0:
            if away_b2b and not home_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=-0.012,
                        summary="Away side is on a tighter turnaround than the opponent.",
                    )
                )
            elif home_b2b and not away_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=0.012,
                        summary="Home side is on a tighter turnaround than the opponent.",
                    )
                )

    if sport in {"basketball", "nhl"}:
        home_b2b = int(_as_float(snapshot.get("home_b2b"), 0))
        away_b2b = int(_as_float(snapshot.get("away_b2b"), 0))
        if direction == 1.0:
            if home_b2b and not away_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=-0.015,
                        summary="Home side is on a tighter turnaround than the opponent.",
                    )
                )
            elif away_b2b and not home_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=0.015,
                        summary="Away side is on a tighter turnaround than the opponent.",
                    )
                )
        elif direction == -1.0:
            if away_b2b and not home_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=-0.015,
                        summary="Away side is on a tighter turnaround than the opponent.",
                    )
                )
            elif home_b2b and not away_b2b:
                adjustments.append(
                    PredictionFactor(
                        name="back_to_back",
                        category="schedule",
                        value=0.015,
                        summary="Home side is on a tighter turnaround than the opponent.",
                    )
                )

    if sport == "basketball":
        home_pace_vs_avg = _as_float(snapshot.get("home_pace_vs_avg"))
        away_pace_vs_avg = _as_float(snapshot.get("away_pace_vs_avg"))
        expected_pace = _as_float(snapshot.get("expected_pace"))
        home_home_wpct = _as_float(snapshot.get("home_home_wpct_20"), 0.5)
        away_away_wpct = _as_float(snapshot.get("away_away_wpct_20"), 0.5)
        home_q4_avg = _as_float(snapshot.get("home_q4_avg"))
        away_q4_avg = _as_float(snapshot.get("away_q4_avg"))
        home_half_ratio = _as_float(snapshot.get("home_half_ratio"), 1.0)
        away_half_ratio = _as_float(snapshot.get("away_half_ratio"), 1.0)
        pace_edge = max(
            -0.012,
            min(0.012, ((home_pace_vs_avg - away_pace_vs_avg) * 0.0018) + ((expected_pace - 225.0) * 0.00018)),
        )
        if direction != 0.0 and pace_edge:
            adjustments.append(
                PredictionFactor(
                    name="pace_control",
                    category="matchup",
                    value=round(pace_edge * direction, 3),
                    summary="Basketball matchup adjustment from pace control and expected tempo.",
                )
            )
        venue_edge = max(-0.01, min(0.01, (home_home_wpct - away_away_wpct) * 0.035))
        if direction != 0.0 and venue_edge:
            adjustments.append(
                PredictionFactor(
                    name="venue_comfort",
                    category="coaching",
                    value=round(venue_edge * direction, 3),
                    summary="Home/away split edge proxy for game-plan comfort and routine execution in this venue context.",
                )
            )
        closing_edge = max(
            -0.01,
            min(0.01, ((home_q4_avg - away_q4_avg) * 0.0022) + ((home_half_ratio - away_half_ratio) * 0.006)),
        )
        if direction != 0.0 and closing_edge:
            adjustments.append(
                PredictionFactor(
                    name="closing_execution",
                    category="coaching",
                    value=round(closing_edge * direction, 3),
                    summary="Late-game execution proxy from Q4 scoring and second-half surge profile.",
                )
            )

        home_injuries = int(_as_float(context.get("home_injuries_count"), 0))
        away_injuries = int(_as_float(context.get("away_injuries_count"), 0))
        home_questionable = int(_as_float(context.get("home_questionable_count"), 0))
        away_questionable = int(_as_float(context.get("away_questionable_count"), 0))
        home_absences = _as_float(context.get("home_rotation_absence_severity"), home_injuries + (home_questionable * 0.5))
        away_absences = _as_float(context.get("away_rotation_absence_severity"), away_injuries + (away_questionable * 0.5))
        home_priority = int(_as_float(context.get("home_priority_absences_count"), home_injuries))
        away_priority = int(_as_float(context.get("away_priority_absences_count"), away_injuries))
        injury_edge = max(-0.02, min(0.02, (away_absences - home_absences) * 0.007))
        if direction != 0.0 and injury_edge:
            adjustments.append(
                PredictionFactor(
                    name="injury_report_edge",
                    category="lineup",
                    value=round(injury_edge * direction, 3),
                    summary="NBA injury report gap from live inactive / questionable feed.",
                )
            )
        priority_edge = max(-0.01, min(0.01, (away_priority - home_priority) * 0.004))
        if direction != 0.0 and priority_edge:
            adjustments.append(
                PredictionFactor(
                    name="rotation_quality_edge",
                    category="lineup",
                    value=round(priority_edge * direction, 3),
                    summary="Selected side has the cleaner core rotation based on confirmed inactive volume.",
                )
            )
        selected_absences = home_absences if direction == 1.0 else away_absences if direction == -1.0 else max(home_absences, away_absences)
        selected_questionable = home_questionable if direction == 1.0 else away_questionable if direction == -1.0 else max(home_questionable, away_questionable)
        if selected_absences >= 2.0 or selected_questionable >= 2:
            adjustments.append(
                PredictionFactor(
                    name="lineup_uncertainty",
                    category="lineup",
                    value=0.0,
                    summary="NBA injury report still has meaningful inactive / questionable volume on one side.",
                )
            )

    if sport == "nhl":
        home_home_wpct = _as_float(snapshot.get("home_home_wpct_20"), 0.5)
        away_away_wpct = _as_float(snapshot.get("away_away_wpct_20"), 0.5)
        home_xg_diff = _first_present(snapshot, "home_xg_diff_10", "home_xg_diff_5")
        away_xg_diff = _first_present(snapshot, "away_xg_diff_10", "away_xg_diff_5")
        home_pp_pct = _first_present(snapshot, "home_pp_pct_10", "home_pp_pct_5", "home_pp_pct")
        away_pp_pct = _first_present(snapshot, "away_pp_pct_10", "away_pp_pct_5", "away_pp_pct")
        home_pk_pct = _first_present(snapshot, "home_pk_pct_10", "home_pk_pct_5", "home_pk_pct")
        away_pk_pct = _first_present(snapshot, "away_pk_pct_10", "away_pk_pct_5", "away_pk_pct")
        special_teams_edge = max(
            -0.012,
            min(0.012, ((home_pp_pct - away_pp_pct) * 0.06) + ((home_pk_pct - away_pk_pct) * 0.05)),
        )
        if direction != 0.0 and special_teams_edge:
            adjustments.append(
                PredictionFactor(
                    name="special_teams_edge",
                    category="matchup",
                    value=round(special_teams_edge * direction, 3),
                    summary="NHL matchup adjustment from power-play and penalty-kill efficiency gap.",
                )
            )
        venue_edge = max(-0.01, min(0.01, (home_home_wpct - away_away_wpct) * 0.03))
        if direction != 0.0 and venue_edge:
            adjustments.append(
                PredictionFactor(
                    name="system_stability",
                    category="coaching",
                    value=round(venue_edge * direction, 3),
                    summary="Home/away split edge proxy for tactical system comfort and bench stability.",
                )
            )
        xg_structure_edge = max(-0.01, min(0.01, (home_xg_diff - away_xg_diff) * 0.004))
        if direction != 0.0 and xg_structure_edge:
            adjustments.append(
                PredictionFactor(
                    name="xg_structure",
                    category="matchup",
                    value=round(xg_structure_edge * direction, 3),
                    summary="Shot-quality structure edge from recent expected-goals differential.",
                )
            )

        home_goalie_confirmed = context.get("home_goalie_confirmed")
        away_goalie_confirmed = context.get("away_goalie_confirmed")
        home_goalie_name = str(context.get("home_goalie_name") or "").strip()
        away_goalie_name = str(context.get("away_goalie_name") or "").strip()
        home_save_pct = _as_float(context.get("home_goalie_save_pct"), 0.0)
        away_save_pct = _as_float(context.get("away_goalie_save_pct"), 0.0)
        home_gaa = _as_float(context.get("home_goalie_gaa"), 0.0)
        away_gaa = _as_float(context.get("away_goalie_gaa"), 0.0)
        if direction == 1.0 and home_goalie_confirmed is not None and not int(home_goalie_confirmed):
            adjustments.append(
                PredictionFactor(
                    name="goalie_uncertainty",
                    category="lineup",
                    value=-0.014,
                    summary="Home goalie is not yet confirmed.",
                )
            )
        elif direction == -1.0 and away_goalie_confirmed is not None and not int(away_goalie_confirmed):
            adjustments.append(
                PredictionFactor(
                    name="goalie_uncertainty",
                    category="lineup",
                    value=-0.014,
                    summary="Away goalie is not yet confirmed.",
                )
            )
        elif direction == 1.0 and home_goalie_confirmed is not None and int(home_goalie_confirmed) and away_goalie_confirmed is not None and not int(away_goalie_confirmed):
            adjustments.append(
                PredictionFactor(
                    name="goalie_stability",
                    category="lineup",
                    value=0.008,
                    summary="Selected side has a confirmed goalie while the opponent still does not.",
                )
            )
        elif direction == -1.0 and away_goalie_confirmed is not None and int(away_goalie_confirmed) and home_goalie_confirmed is not None and not int(home_goalie_confirmed):
            adjustments.append(
                PredictionFactor(
                    name="goalie_stability",
                    category="lineup",
                    value=0.008,
                    summary="Selected side has a confirmed goalie while the opponent still does not.",
                )
            )
        if direction == 1.0 and home_goalie_name and not away_goalie_name:
            adjustments.append(
                PredictionFactor(
                    name="probable_goalie_named",
                    category="lineup",
                    value=0.004,
                    summary="Selected side has a named probable goalie while the opponent listing is still vague.",
                )
            )
        elif direction == -1.0 and away_goalie_name and not home_goalie_name:
            adjustments.append(
                PredictionFactor(
                    name="probable_goalie_named",
                    category="lineup",
                    value=0.004,
                    summary="Selected side has a named probable goalie while the opponent listing is still vague.",
                )
            )
        goalie_quality_edge = 0.0
        if home_save_pct and away_save_pct:
            goalie_quality_edge += max(-0.012, min(0.012, (home_save_pct - away_save_pct) * 0.18))
        if home_gaa and away_gaa:
            goalie_quality_edge += max(-0.01, min(0.01, (away_gaa - home_gaa) * 0.012))
        if direction != 0.0 and goalie_quality_edge:
            adjustments.append(
                PredictionFactor(
                    name="goalie_quality",
                    category="lineup",
                    value=round(goalie_quality_edge * direction, 3),
                    summary="Named goalie quality edge from save percentage and goals-against profile.",
                )
            )

    return adjustments
