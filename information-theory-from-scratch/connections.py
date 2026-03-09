# The information-theoretic view of machine learning

connections = [
    ("Classification",     "Cross-entropy loss",     "min D_KL(p_true || q_model)"),
    ("VAEs",               "ELBO",                   "min H(p,q) + D_KL(q(z|x)||p(z))"),
    ("Diffusion Models",   "Noise prediction",       "min cross-entropy in noise space"),
    ("Contrastive (CL)",   "InfoNCE",                "max MI lower bound"),
    ("Decision Trees",     "Information gain",        "max MI(feature, label)"),
    ("RLHF / DPO",        "Reward + KL penalty",    "max R - beta * D_KL(pi||pi_ref)"),
    ("Language Models",    "Next-token prediction",  "min per-token H(p,q) = log(PPL)"),
]

print(f"{'Algorithm':<22} {'Loss/Objective':<24} {'Info-Theoretic Form'}")
print("-" * 78)
for algo, loss, info in connections:
    print(f"{algo:<22} {loss:<24} {info}")

print("\nOne language. One framework. Every loss function is about")
print("minimizing surprise — the core insight of information theory.")
