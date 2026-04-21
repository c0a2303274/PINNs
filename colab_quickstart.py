"""Minimal Colab bootstrap notes.

Copy the cells below into a Colab notebook after pushing this repo to GitHub.
"""

COLAB_CELLS = r"""
# Cell 1: clone the repository
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_NAME>

# Cell 2: optional package install
!pip install -r requirements.txt

# Cell 3: train the baseline Poisson PINN
!python train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --output-dir outputs/poisson_baseline_1k

# Cell 4: append a research log entry after the run
!python append_research_log.py --stage poisson-baseline --pde Poisson --setting "Omega=[-1,1]^2, epochs=1000, n_interior=1024, n_boundary=256" --changes "Ran baseline on Colab" --done "Completed baseline GPU run" --result "Inspect metrics.json and plots in outputs/poisson_baseline_1k" --next-step "Compare runtime and error with local run"
"""


if __name__ == "__main__":
    print(COLAB_CELLS.strip())
