from pathlib import Path

def plot_and_save(dir_: Path, name: str, fig):
    fig.show()

    path = dir_ / f'{name}.png'
    if not path.exists():
        dir_.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, transparent=False, dpi=80, bbox_inches="tight")
