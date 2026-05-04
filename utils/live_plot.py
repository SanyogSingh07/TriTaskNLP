import matplotlib.pyplot as plt
import os

class LivePlot:
    def __init__(self):
        plt.ion()  # interactive mode ON

        try:
            from utils.visualization import apply_nvidia_theme
            apply_nvidia_theme()
        except ImportError:
            pass

        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        self.epochs = []
        self.topic_losses = []
        self.sent_losses = []
        self.auth_losses = []

        self.line_t, = self.ax.plot([], [], label="Topic Loss", color="#1a4d00", marker="o", linewidth=2)
        self.line_s, = self.ax.plot([], [], label="Sentiment Loss", color="#76B900", marker="x", linewidth=2)
        self.line_a, = self.ax.plot([], [], label="Author Loss", color="#d1ff66", marker="^", linewidth=2)

        self.ax.set_title("Training Loss (Live)", color="#76B900", fontweight="bold")
        self.ax.set_xlabel("Epoch", color="#76B900")
        self.ax.set_ylabel("Loss", color="#76B900")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(facecolor='#0b0b0b', edgecolor='#76B900', labelcolor='white')

    def update(self, epoch, t_loss, s_loss, a_loss):
        self.epochs.append(epoch)
        self.topic_losses.append(t_loss)
        self.sent_losses.append(s_loss)
        self.auth_losses.append(a_loss)

        self.line_t.set_xdata(self.epochs)
        self.line_t.set_ydata(self.topic_losses)
        
        self.line_s.set_xdata(self.epochs)
        self.line_s.set_ydata(self.sent_losses)
        
        self.line_a.set_xdata(self.epochs)
        self.line_a.set_ydata(self.auth_losses)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.1)

    def save(self):
        os.makedirs("plots", exist_ok=True)
        save_path = os.path.abspath("plots/live_training_loss.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.ioff()
        plt.close()
