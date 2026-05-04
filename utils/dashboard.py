from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

import torch
import os

console = Console()

def create_dashboard():
    table = Table(title="🧠 TriTaskNLP Training Dashboard")

    table.add_column("Epoch", justify="center")
    table.add_column("Loss", justify="center")
    table.add_column("GPU Memory", justify="center")
    table.add_column("Status", justify="center")

    return table

def get_gpu_usage():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    return "CPU"

def train_with_dashboard(model, loader, optimizer, criterion, epochs, device, vocab, maps):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Loss balance weights: λ_topic, λ_sentiment, λ_author
    lambda_weights = [0.5, 0.3, 0.2]

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    task = progress.add_task("Training...", total=len(loader) * epochs)

    dashboard = create_dashboard()
    
    from utils.live_plot import LivePlot
    plotter = LivePlot()

    with Live(Panel(dashboard), refresh_per_second=4) as live:
        for epoch in range(epochs):
            total_loss = 0
            epoch_t_loss = 0
            epoch_s_loss = 0
            epoch_a_loss = 0
            
            model.train()

            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                stylo = batch['stylo'].to(device)
                y_topic = batch['topic'].to(device)
                y_sent = batch['sentiment'].to(device)
                y_auth = batch['author'].to(device)

                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        t_out, s_out, a_out, _ = model(input_ids, stylo)
                        loss_t = criterion(t_out, y_topic)
                        loss_s = criterion(s_out, y_sent)
                        loss_a = criterion(a_out, y_auth)
                        loss = lambda_weights[0]*loss_t + lambda_weights[1]*loss_s + lambda_weights[2]*loss_a
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    t_out, s_out, a_out, _ = model(input_ids, stylo)
                    loss_t = criterion(t_out, y_topic)
                    loss_s = criterion(s_out, y_sent)
                    loss_a = criterion(a_out, y_auth)
                    loss = lambda_weights[0]*loss_t + lambda_weights[1]*loss_s + lambda_weights[2]*loss_a
                    
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                epoch_t_loss += loss_t.item()
                epoch_s_loss += loss_s.item()
                epoch_a_loss += loss_a.item()

                progress.advance(task)

            num_batches = len(loader)
            avg_loss = total_loss / num_batches
            avg_t = epoch_t_loss / num_batches
            avg_s = epoch_s_loss / num_batches
            avg_a = epoch_a_loss / num_batches

            dashboard.add_row(
                str(epoch + 1),
                f"{avg_loss:.4f}",
                get_gpu_usage(),
                "Completed"
            )

            # Real-time Live Stats Panel
            live.update(
                Panel(
                    f"{dashboard}\n\n[bold green]Live Stats[/bold green]\nEpoch: {epoch+1}\nLoss: {avg_loss:.4f}\nGPU: {get_gpu_usage()}",
                    title="TriTaskNLP Live Training"
                )
            )
            
            # LIVE GRAPH UPDATE
            plotter.update(epoch + 1, avg_t, avg_s, avg_a)

    plotter.save()
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'maps': maps
    }, "models/model.pth")
    console.print("\nTraining Complete & Model Saved!", style="bold green")
