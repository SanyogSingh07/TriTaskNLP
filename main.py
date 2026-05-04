import typer
import sys
import os
import multiprocessing
import time

# Bypass joblib/loky's wmic core counting which crashes on Windows cp1252
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count() or 1)
try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores_win32 = lambda: multiprocessing.cpu_count() or 1
except Exception:
    pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.align import Align

from train import train
from evaluate import evaluate
from cli.commands import predict
from utils.visualization import plot_all
from utils.metrics import plot_confusion_matrix
from utils.init_system import initialize_system, state

import numpy as np
import warnings

# Force UTF-8 encoding for Windows terminals to support emojis
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except Exception:
        pass

app = typer.Typer()
console = Console()

initialized = False

def initialize_once():
    global initialized
    if not initialized:
        initialize_system()
        initialized = True

# -----------------------------
# Banner
# -----------------------------
def show_banner():
    banner = Panel.fit(
        "[bold green]TriTaskNLP Dashboard[/bold green]\n"
        "[dim]Topic • Sentiment • Author Identification[/dim]",
        border_style="green"
    )
    console.print(banner)

def animated_step(message, delay=1.2):
    with console.status(f"[bold green]{message}...", spinner="dots"):
        time.sleep(delay)

def screen_transition():
    console.clear()
    time.sleep(0.3)

# -----------------------------
# INTERACTIVE MENU
# -----------------------------
def interactive_menu():
    while True:
        screen_transition()
        show_banner()

        console.print("\n[bold green]Main Menu[/bold green]\n")

        menu = Table.grid(padding=1)
        menu.add_column()
        menu.add_column()

        menu.add_row("1. Train Model", "2. Evaluate Model")
        menu.add_row("3. Predict Text", "4. Visualize Data")
        menu.add_row("5. Show Dashboard", "0. Exit")

        console.print(menu)

        choice = console.input("\n> Enter your choice [0-5]: ")

        # -----------------------------
        # TRAIN
        # -----------------------------
        if choice == "1":
            screen_transition()
            show_banner()

            animated_step("Initializing system")
            initialize_once()

            animated_step("Loading model")
            animated_step("Preparing dataset")

            console.print("\n[bold green]Training Model...\n")
            train()

            console.print("\n[bold green]Training Completed Successfully![/bold green]\n")
            input("Press Enter to continue...")

        # -----------------------------
        # EVALUATE
        # -----------------------------
        elif choice == "2":
            screen_transition()
            show_banner()

            animated_step("Initializing system")
            initialize_once()

            animated_step("Running evaluation")

            results = evaluate()

            if results:
                table = Table(title="Model Performance")
                table.add_column("Metric")
                table.add_column("Value")

                for k, v in results.items():
                    table.add_row(k, v)

                console.print(table)
            
            # The real evaluate() function already generates and saves the confusion matrix

            console.print("\nEvaluation Completed!\n")
            input("Press Enter to continue...")

        # -----------------------------
        # PREDICT
        # -----------------------------
        elif choice == "3":
            screen_transition()
            show_banner()

            text = console.input("\n> Enter text: ")

            if text.strip():
                animated_step("Initializing system")
                initialize_once()
                animated_step("Running inference")

                result = predict(text)

                if result and "Error" not in result:
                    table = Table(title="Prediction Output")
                    table.add_column("Task")
                    table.add_column("Result")

                    for k, v in result.items():
                        table.add_row(k, v)

                    console.print(table)
            else:
                console.print("[yellow]No input provided.[/yellow]")

            input("\nPress Enter to continue...")

        # -----------------------------
        # VISUALIZE
        # -----------------------------
        elif choice == "4":
            screen_transition()
            show_banner()

            animated_step("Initializing system")
            initialize_once()
            animated_step("Preparing embeddings")
            animated_step("Generating plots")

            plot_all()

            console.print("\nVisualizations Complete!\n")
            input("Press Enter to continue...")

        # -----------------------------
        # DASHBOARD
        # -----------------------------
        elif choice == "5":
            screen_transition()
            dashboard()
            input("\nPress Enter to continue...")

        # -----------------------------
        # EXIT
        # -----------------------------
        elif choice == "0":
            console.print("\nExiting TriTaskNLP. Goodbye!\n")
            break

        else:
            console.print("\n[red]Invalid choice. Try again.[/red]\n")
            time.sleep(1)

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """TriTaskNLP CLI"""
    if ctx.invoked_subcommand is None:
        interactive_menu()

# -----------------------------
# TRAIN
# -----------------------------
@app.command(name="train")
def train_cmd():
    show_banner()
    initialize_once()
    console.print("\n[bold green]Training Model...[/bold green]\n")
    train()
    console.print("\n[bold green]Training Completed Successfully![/bold green]\n")

# -----------------------------
# EVALUATE
# -----------------------------
@app.command(name="evaluate")
def evaluate_cmd():
    show_banner()
    initialize_once()
    console.print("\n[bold cyan]Evaluating Model...[/bold cyan]\n")

    results = evaluate()  

    if not results:
        return

    # Table Output
    table = Table(title="Model Performance")
    table.add_column("Metric", justify="center")
    table.add_column("Value", justify="center")

    for key, value in results.items():
        table.add_row(key, value)

    console.print(table)
    console.print("\nEvaluation Completed!\n")


# -----------------------------
# PREDICT
# -----------------------------
@app.command(name="predict")
def predict_cmd(text: str = typer.Argument(...)):
    show_banner()
    initialize_once()

    console.print("\n[bold yellow]Prediction Result[/bold yellow]\n")

    result = predict(text)

    if not result or "Error" in result:
        return

    table = Table(title="Prediction Output")
    table.add_column("Task")
    table.add_column("Result")

    for k, v in result.items():
        table.add_row(k, v)

    console.print(table)

# -----------------------------
# VISUALIZE
# -----------------------------
@app.command(name="visualize")
def visualize():
    show_banner()
    initialize_once()

    console.print("\n[bold magenta]Generating Visual Dashboard...[/bold magenta]\n")

    plot_all()

    console.print("\nVisualizations Complete!\n")

# -----------------------------
# DASHBOARD
# -----------------------------
@app.command(name="dashboard")
def dashboard():
    show_banner()

    console.print("\n[bold green]Full System Dashboard[/bold green]\n")

    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column()

    grid.add_row(
        Panel("Train Model\n[dim]python TriTaskNLP\\main.py train[/dim]", border_style="green"),
        Panel("Evaluate Model\n[dim]python TriTaskNLP\\main.py evaluate[/dim]", border_style="cyan"),
    )

    grid.add_row(
        Panel("Predict Text\n[dim]python TriTaskNLP\\main.py predict \"text\"[/dim]", border_style="yellow"),
        Panel("Visualizations\n[dim]python TriTaskNLP\\main.py visualize[/dim]", border_style="magenta"),
    )

    console.print(grid)

# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    app()
