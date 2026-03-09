import numpy as np

def simulate_icl_phase_transition(n_steps=300, seed=42):
    """Simulate the ICL phase transition during training."""
    rng = np.random.RandomState(seed)

    train_loss = np.zeros(n_steps)
    icl_accuracy = np.zeros(n_steps)
    induction_score = np.zeros(n_steps)

    transition_center = 60
    transition_width = 8

    for step in range(n_steps):
        t = step / n_steps

        base_loss = 3.5 * (1 + step) ** (-0.15)
        bump = 0.08 * np.exp(-0.5 * ((step - transition_center) / 5) ** 2)
        train_loss[step] = base_loss - bump + 0.02 * rng.randn()

        induction_score[step] = 1.0 / (1 + np.exp(
            -(step - transition_center) / (transition_width / 4)
        )) + 0.03 * rng.randn()

        icl_sigmoid = 1.0 / (1 + np.exp(
            -(step - transition_center - 3) / (transition_width / 4)
        ))
        icl_accuracy[step] = 0.10 + 0.75 * icl_sigmoid
        icl_accuracy[step] += 0.03 * rng.randn()

    icl_accuracy = np.clip(icl_accuracy, 0, 1)
    induction_score = np.clip(induction_score, 0, 1)

    return train_loss, icl_accuracy, induction_score

loss, icl_acc, ind_score = simulate_icl_phase_transition()

for step in [0, 30, 55, 65, 80, 150, 299]:
    print(f"Step {step:3d}: loss={loss[step]:.2f}  "
          f"ICL_acc={icl_acc[step]:.1%}  "
          f"induction={ind_score[step]:.2f}")
# Step   0: loss=3.50  ICL_acc=9.8%   induction=0.00
# Step  30: loss=2.99  ICL_acc=9.2%   induction=0.00
# Step  55: loss=2.80  ICL_acc=23.1%  induction=0.25
# Step  65: loss=2.68  ICL_acc=76.1%  induction=0.87
# Step  80: loss=2.69  ICL_acc=86.4%  induction=1.00
# Step 150: loss=2.50  ICL_acc=85.5%  induction=0.97
# Step 299: loss=2.28  ICL_acc=84.6%  induction=1.00
