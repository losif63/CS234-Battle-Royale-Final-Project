# Battle Royale

A simple 2-player local battle royale: move around the arena, pick up ammo (green circles), and shoot the other player. First hit wins.

## How to run

From the project root:

```bash
uv run python -m src.main
```

To watch two random agents play instead:

```bash
uv run python -m agents.random_agent --render
```

## Keybindings

|        | Player 1 | Player 2 |
|--------|----------|----------|
| Move   | **W** up, **S** down, **A** left, **D** right | **Arrow keys** |
| Shoot  | **R** up, **F** down, **G** left, **H** right | **I** up, **K** down, **J** left, **L** right |

- **ESC** or close the window to quit.
- Shoot only fires once per key press (tap to shoot, not hold).
