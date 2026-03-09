from soft_labels import softmax, teacher_logits

# Same teacher logits from above — let's see what happens
# as we raise the temperature

for T in [1, 2, 5, 10, 20]:
    probs = softmax(teacher_logits, T=T)
    # Show only the interesting digits: 2 (correct), 3, 7 (dark knowledge)
    print(f"T={T:2d}:  P(2)={probs[2]:.4f}  P(3)={probs[3]:.4f}  "
          f"P(7)={probs[7]:.4f}  P(0)={probs[0]:.4f}")

# T= 1:  P(2)=0.8846  P(3)=0.0487  P(7)=0.0295  P(0)=0.0013
# T= 2:  P(2)=0.5189  P(3)=0.1217  P(7)=0.0948  P(0)=0.0201
# T= 5:  P(2)=0.2231  P(3)=0.1249  P(7)=0.1130  P(0)=0.0608
# T=10:  P(2)=0.1524  P(3)=0.1140  P(7)=0.1085  P(0)=0.0796
# T=20:  P(2)=0.1240  P(3)=0.1073  P(7)=0.1046  P(0)=0.0896
