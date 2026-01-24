import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def generate_gif(env, agent, output_path="replay.gif", max_steps=300):
    """Runs an episode and saves a GIF animation."""
    frames = []
    state, _ = env.reset()
    done = False
    step = 0
    
    # Render initial state
    frames.append(_render_frame(state, env.current_phase, step))
    
    while not done and step < max_steps:
        action = agent.act(state, training=False)
        state, _, done, truncated, _ = env.step(action)
        done = done or truncated
        step += 1
        frames.append(_render_frame(state, env.current_phase, step))
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=5)
    return output_path

def _render_frame(state, phase, step):
    """Draws the intersection using Matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.axis('off')
    
    # Roads
    ax.fill_between([-3, 3], -12, 12, color='#444444', alpha=0.5)
    ax.fill_between([-12, 12], -3, 3, color='#444444', alpha=0.5)
    
    # Queues
    queues = state[:12].astype(int)
    bbox = dict(boxstyle='round', facecolor='white', alpha=0.8)
    
    ax.text(0, 8, f"N: {queues[0:3]}", ha='center', bbox=bbox)
    ax.text(0, -8, f"S: {queues[3:6]}", ha='center', bbox=bbox)
    ax.text(8, 0, f"E: {queues[6:9]}", va='center', bbox=bbox)
    ax.text(-8, 0, f"W: {queues[9:12]}", va='center', bbox=bbox)

    phase_names = {0: "NS Green", 1: "NS Left", 2: "EW Green", 3: "EW Left"}
    color = 'green' if phase in [0, 2] else 'orange'
    ax.text(0, 0, f"{phase_names.get(phase, '?')}\nStep: {step}", 
            ha='center', va='center', fontweight='bold', color=color, 
            bbox=dict(facecolor='black', alpha=0.1))

    # --- FIX FOR MATPLOTLIB ERROR ---
    fig.canvas.draw()
    # Get RGBA buffer and convert to numpy
    image = np.asarray(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB (drop alpha channel)
    if image.shape[2] == 4:
        image = image[:, :, :3]
        
    plt.close(fig)
    return image