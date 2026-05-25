#!/usr/bin/env python3
"""
Sports Predictor - CLI Entry Point.

Full pipeline: fetch data → engineer features → train models →
predict upcoming matches → detect value bets → manage bankroll.

Usage:
    python main.py pipeline --sport soccer
    python main.py fetch --sport basketball
    python main.py train --sport tennis
    python main.py predict --sport soccer
    python main.py backtest --sport soccer
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from src.utils.logger import setup_logger
from src.data.soccer_fetcher import SoccerFetcher
from src.data.basketball_fetcher import BasketballFetcher
from src.data.tennis_fetcher import TennisFetcher
from src.data.odds_fetcher import OddsFetcher
from src.features.soccer_features import SoccerFeatureEngineer
from src.features.basketball_features import BasketballFeatureEngineer
from src.features.tennis_features import TennisFeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.predictor import Predictor
from src.models.calibration import ProbabilityCalibrator
from src.risk.kelly import KellyCriterion
from src.risk.bankroll import BankrollManager, Bet
from src.risk.value_detector import ValueDetector
from src.evaluation.backtester import Backtester
from src.evaluation.metrics import MetricsCalculator

logger = setup_logger(__name__)
console = Console()

# ------------------------------------------------------------------
# Sport Registry
# ------------------------------------------------------------------

FETCHERS = {
    "soccer": SoccerFetcher,
    "basketball": BasketballFetcher,
    "tennis": TennisFetcher,
}

FEATURE_ENGINEERS = {
    "soccer": SoccerFeatureEngineer,
    "basketball": BasketballFeatureEngineer,
    "tennis": TennisFeatureEngineer,
}


def _get_enabled_sports() -> list[str]:
    """Return list of sports enabled in config."""
    sports_cfg = settings.get("sports", {})
    return [s for s, cfg in sports_cfg.items() if cfg.get("enabled", False)]


# ------------------------------------------------------------------
# CLI Group
# ------------------------------------------------------------------

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """Sports Predictor - ML Betting Prediction System."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------
# FETCH Command
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis", "all"]),
              default="all", help="Sport to fetch data for")
@click.option("--season", default=None, help="Specific season (e.g. '2024')")
def fetch(sport: str, season: Optional[str]) -> None:
    """Fetch match data and odds from APIs."""
    sports = _get_enabled_sports() if sport == "all" else [sport]

    for s in sports:
        console.print(f"\n[bold blue]Fetching {s} data...[/bold blue]")

        try:
            fetcher_cls = FETCHERS.get(s)
            if not fetcher_cls:
                console.print(f"[red]No fetcher for sport: {s}[/red]")
                continue

            fetcher = fetcher_cls()

            if season:
                df = fetcher.fetch_matches(season=season)
            else:
                df = fetcher.fetch_all_seasons()

            if df.empty:
                console.print(f"[yellow]No data retrieved for {s}[/yellow]")
                continue

            fetcher.save_raw(df, f"{s}_matches")
            console.print(f"[green]✓ {s}: {len(df)} matches fetched[/green]")

            # Fetch odds
            console.print(f"[blue]Fetching {s} odds...[/blue]")
            odds_fetcher = OddsFetcher(sport=s, cache_expire_hours=1)
            odds_df = odds_fetcher.fetch_odds()

            if not odds_df.empty:
                odds_fetcher.save_raw(odds_df, f"{s}_odds")
                console.print(f"[green]✓ {s}: {len(odds_df)} odds entries fetched[/green]")

        except Exception as exc:
            console.print(f"[red]Error fetching {s}: {exc}[/red]")
            logger.exception("Fetch error for %s", s)


# ------------------------------------------------------------------
# FEATURES Command
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis", "all"]),
              default="all")
def features(sport: str) -> None:
    """Engineer features from raw match data."""
    sports = _get_enabled_sports() if sport == "all" else [sport]

    for s in sports:
        console.print(f"\n[bold blue]Engineering {s} features...[/bold blue]")

        try:
            # Load raw data
            fetcher = FETCHERS[s]()
            raw_df = fetcher.load_processed(f"{s}_matches")
            if raw_df is None:
                raw_path = Path(settings["paths"]["raw_data"]) / s / f"{s}_matches.parquet"
                if raw_path.exists():
                    import pandas as pd
                    raw_df = pd.read_parquet(raw_path)
                else:
                    console.print(f"[yellow]No raw data for {s}. Run 'fetch' first.[/yellow]")
                    continue

            # Engineer features
            engineer = FEATURE_ENGINEERS[s]()
            featured_df = engineer.engineer_features(raw_df)

            if featured_df.empty:
                console.print(f"[yellow]Feature engineering produced empty result for {s}[/yellow]")
                continue

            # Save
            fetcher.save_processed(featured_df, f"{s}_featured")
            console.print(
                f"[green]✓ {s}: {featured_df.shape[1]} features × "
                f"{len(featured_df)} matches[/green]"
            )

        except Exception as exc:
            console.print(f"[red]Error engineering {s} features: {exc}[/red]")
            logger.exception("Feature engineering error for %s", s)


# ------------------------------------------------------------------
# TRAIN Command
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis", "all"]),
              default="all")
@click.option("--cv/--no-cv", default=True, help="Run cross-validation before training")
@click.option("--tag", default="latest", help="Model version tag")
def train(sport: str, cv: bool, tag: str) -> None:
    """Train prediction models."""
    sports = _get_enabled_sports() if sport == "all" else [sport]

    for s in sports:
        console.print(f"\n[bold blue]Training {s} models...[/bold blue]")

        try:
            import pandas as pd

            # Load featured data
            processed_path = Path(settings["paths"]["processed_data"]) / s / f"{s}_featured.parquet"
            if not processed_path.exists():
                console.print(f"[yellow]No featured data for {s}. Run 'features' first.[/yellow]")
                continue

            df = pd.read_parquet(processed_path)
            engineer = FEATURE_ENGINEERS[s]()
            X, y = engineer.prepare_for_training(df)

            if len(X) == 0:
                console.print(f"[yellow]No training data for {s}[/yellow]")
                continue

            # Train/test split (last 20% as holdout)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            trainer = ModelTrainer(sport=s)

            # Cross-validation
            if cv:
                console.print("[blue]Running cross-validation...[/blue]")
                cv_results = trainer.cross_validate(X_train, y_train)

                cv_table = Table(title=f"{s.title()} CV Results")
                cv_table.add_column("Algorithm")
                cv_table.add_column("Accuracy")
                cv_table.add_column("Log Loss")
                cv_table.add_column("Folds")

                for name, metrics in cv_results.items():
                    cv_table.add_row(
                        name,
                        f"{metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}",
                        f"{metrics['mean_log_loss']:.4f} ± {metrics['std_log_loss']:.4f}",
                        str(metrics["n_folds"]),
                    )
                console.print(cv_table)

            # Full training
            console.print("[blue]Training on full training set...[/blue]")
            trainer.train(X_train, y_train, X_test, y_test)

            # Build ensemble
            console.print("[blue]Building ensemble...[/blue]")
            trainer.build_ensemble(X_train, y_train)

            # Evaluate on test set
            test_results = trainer.evaluate(X_test, y_test)

            results_table = Table(title=f"{s.title()} Test Results")
            results_table.add_column("Model")
            results_table.add_column("Accuracy")
            results_table.add_column("Log Loss")

            for name, metrics in test_results.items():
                results_table.add_row(
                    name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['log_loss']:.4f}",
                )
            console.print(results_table)

            # Calibrate
            console.print("[blue]Calibrating probabilities...[/blue]")
            cal_path = Path(settings["paths"]["models"]) / s / f"calibrator_{tag}.joblib"
            best_model = trainer.ensemble_model or list(trainer.trained_models.values())[0]

            # Hold back part of the test window for calibration, and only keep
            # the calibrator if it improves held-out metrics on the remainder.
            if len(X_test) < 40:
                console.print(f"[yellow]Skipping calibration for {s}: test set too small.[/yellow]")
                if cal_path.exists():
                    cal_path.unlink()
            else:
                split_cal = max(20, int(len(X_test) * 0.5))
                if split_cal >= len(X_test):
                    split_cal = len(X_test) - 1

                X_cal = X_test.iloc[:split_cal]
                y_cal = y_test.iloc[:split_cal]
                X_eval = X_test.iloc[split_cal:]
                y_eval = y_test.iloc[split_cal:]

                if len(X_eval) < 10:
                    console.print(f"[yellow]Skipping calibration for {s}: evaluation slice too small.[/yellow]")
                    if cal_path.exists():
                        cal_path.unlink()
                else:
                    calibrator = ProbabilityCalibrator()
                    calibrator.fit(best_model, X_cal, y_cal.values)
                    cal_metrics = calibrator.evaluate(best_model, X_eval, y_eval.values)

                    if (
                        cal_metrics["log_loss_improvement"] > 0
                        and cal_metrics["brier_improvement"] > 0
                    ):
                        calibrator.save(cal_path)
                        console.print(
                            f"[green]✓ Calibration kept for {s} "
                            f"(log_loss Δ{cal_metrics['log_loss_improvement']:+.4f}, "
                            f"brier Δ{cal_metrics['brier_improvement']:+.4f})[/green]"
                        )
                    else:
                        if cal_path.exists():
                            cal_path.unlink()
                        console.print(
                            f"[yellow]Calibration skipped for {s}: "
                            f"held-out metrics did not improve "
                            f"(log_loss Δ{cal_metrics['log_loss_improvement']:+.4f}, "
                            f"brier Δ{cal_metrics['brier_improvement']:+.4f}).[/yellow]"
                        )

            # Save models
            paths = trainer.save_models(tag=tag)
            console.print(f"[green]✓ {s}: Models saved ({len(paths)} files)[/green]")

        except Exception as exc:
            console.print(f"[red]Error training {s}: {exc}[/red]")
            logger.exception("Training error for %s", s)


# ------------------------------------------------------------------
# PREDICT Command
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis", "all"]),
              default="all")
@click.option("--tag", default="latest")
def predict(sport: str, tag: str) -> None:
    """Predict upcoming matches and detect value bets."""
    sports = _get_enabled_sports() if sport == "all" else [sport]

    for s in sports:
        console.print(f"\n[bold blue]Predicting {s} matches...[/bold blue]")

        try:
            import pandas as pd

            # Load featured data (for label map)
            processed_path = Path(settings["paths"]["processed_data"]) / s / f"{s}_featured.parquet"
            if not processed_path.exists():
                console.print(f"[yellow]No featured data for {s}[/yellow]")
                continue

            df = pd.read_parquet(processed_path)
            engineer = FEATURE_ENGINEERS[s]()

            # Get label map from the data
            if "result" in df.columns:
                labels = sorted(df["result"].dropna().unique())
                label_map = {label: idx for idx, label in enumerate(labels)}
            else:
                console.print(f"[yellow]No result column for {s}[/yellow]")
                continue

            # Load predictor
            predictor = Predictor(sport=s, label_map=label_map, tag=tag)

            # Fetch current odds
            odds_fetcher = OddsFetcher(sport=s, cache_expire_hours=1)
            odds_df = odds_fetcher.fetch_odds()

            if odds_df.empty:
                console.print(f"[yellow]No current odds available for {s}[/yellow]")
                continue

            best_odds = odds_fetcher.get_best_odds(odds_df)

            # For prediction, we'd need upcoming match features
            # Here we demonstrate on the most recent data
            X, y = engineer.prepare_for_training(df)
            recent_X = X.tail(20)

            predictions = predictor.predict_proba(recent_X)
            console.print(f"\n[bold]Predictions for {s}:[/bold]")
            console.print(predictions.round(3).to_string())

            # Value detection
            value_detector = ValueDetector()
            value_bets = value_detector.detect(predictions, best_odds)

            if value_bets:
                vb_df = value_detector.to_dataframe(value_bets)

                vb_table = Table(title=f"{s.title()} Value Bets")
                vb_table.add_column("Match")
                vb_table.add_column("Outcome")
                vb_table.add_column("Odds")
                vb_table.add_column("Edge %")
                vb_table.add_column("EV %")
                vb_table.add_column("Kelly %")

                for _, row in vb_df.iterrows():
                    vb_table.add_row(
                        f"{row['home_team']} vs {row['away_team']}",
                        row["outcome"],
                        f"{row['best_odds']:.2f}",
                        f"{row['edge_pct']:.1f}%",
                        f"{row['ev_pct']:.1f}%",
                        f"{row['kelly_stake'] * 100:.2f}%",
                    )
                console.print(vb_table)
            else:
                console.print(f"[yellow]No value bets detected for {s}[/yellow]")

        except Exception as exc:
            console.print(f"[red]Error predicting {s}: {exc}[/red]")
            logger.exception("Prediction error for %s", s)


# ------------------------------------------------------------------
# BACKTEST Command
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis"]),
              required=True)
@click.option("--bankroll", default=1000.0, help="Starting bankroll")
def backtest(sport: str, bankroll: float) -> None:
    """Run walk-forward backtest on historical data."""
    console.print(f"\n[bold blue]Backtesting {sport}...[/bold blue]")

    try:
        import pandas as pd

        processed_path = Path(settings["paths"]["processed_data"]) / sport / f"{sport}_featured.parquet"
        if not processed_path.exists():
            console.print(f"[yellow]No featured data. Run 'features' first.[/yellow]")
            return

        df = pd.read_parquet(processed_path)
        engineer = FEATURE_ENGINEERS[sport]()

        bt = Backtester(
            sport=sport,
            feature_engineer=engineer,
            initial_bankroll=bankroll,
        )

        results = bt.run(df)

        # Display results
        console.print(Panel.fit(
            f"[bold]Backtest Results: {sport.title()}[/bold]\n\n"
            f"Periods: {results.get('n_periods', 0)}\n"
            f"Final Bankroll: ${results.get('final_bankroll', 0):.2f}\n"
            f"Max Drawdown: {results.get('max_drawdown', 0) * 100:.1f}%\n"
            f"Overall Accuracy: {results.get('overall_accuracy', 0) * 100:.1f}%",
            title="Summary",
            border_style="green",
        ))

        # Period summary
        summary_df = bt.get_summary_df()
        if not summary_df.empty:
            console.print("\n[bold]Period Breakdown:[/bold]")
            console.print(summary_df.to_string(index=False))

        # Bankroll stats
        stats = results.get("bankroll_stats", {})
        if stats:
            stats_table = Table(title="Bankroll Statistics")
            stats_table.add_column("Metric")
            stats_table.add_column("Value")
            for k, v in stats.items():
                if isinstance(v, float):
                    stats_table.add_row(k, f"{v:.4f}")
                else:
                    stats_table.add_row(k, str(v))
            console.print(stats_table)

    except Exception as exc:
        console.print(f"[red]Backtest error: {exc}[/red]")
        logger.exception("Backtest error for %s", sport)


# ------------------------------------------------------------------
# PIPELINE Command (full end-to-end)
# ------------------------------------------------------------------

@cli.command()
@click.option("--sport", type=click.Choice(["soccer", "basketball", "tennis", "all"]),
              default="all")
@click.option("--tag", default="latest")
def pipeline(sport: str, tag: str) -> None:
    """Run full pipeline: fetch → features → train → predict."""
    console.print("[bold magenta]Running full pipeline...[/bold magenta]\n")

    # Invoke sub-commands programmatically
    ctx = click.get_current_context()

    console.print("[bold]Step 1/4: Fetching data[/bold]")
    ctx.invoke(fetch, sport=sport, season=None)

    console.print("\n[bold]Step 2/4: Engineering features[/bold]")
    ctx.invoke(features, sport=sport)

    console.print("\n[bold]Step 3/4: Training models[/bold]")
    ctx.invoke(train, sport=sport, cv=True, tag=tag)

    console.print("\n[bold]Step 4/4: Generating predictions[/bold]")
    ctx.invoke(predict, sport=sport, tag=tag)

    console.print("\n[bold green]Pipeline complete![/bold green]")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    cli()
