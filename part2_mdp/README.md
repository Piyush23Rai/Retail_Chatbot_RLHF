# PART 2: MARKOV DECISION PROCESS (MDP) FORMULATION

## 1. Objective

The objective of Part 2 is to model a **retail customer–chatbot interaction** as a **Markov Decision Process (MDP)**. The goal of the chatbot agent is to **maximize cumulative customer satisfaction and conversion probability** over the course of an interaction.

This formulation allows us to mathematically reason about sequential decision-making under uncertainty and evaluate chatbot policies using reinforcement learning concepts.

---

## 2. Markov Decision Process Overview

An MDP is defined as a 5-tuple:

[ \langle S, A, P, R, \gamma \rangle ]

Where:

* **S**: State space
* **A**: Action space
* **P**: Transition dynamics
* **R**: Reward function
* **γ**: Discount factor

In this assignment, each component is explicitly defined and implemented.

---

## 3. State Space (S)

The state represents the **current interaction context** between the customer and the chatbot.

### State Variables

| Variable              | Description                  | Type                                  |
| --------------------- | ---------------------------- | ------------------------------------- |
| customer_segment      | Customer type                | {new, regular, vip}                   |
| product_category      | Product category of interest | {Electronics, Clothing, Home, Sports} |
| customer_satisfaction | Predicted satisfaction level | Continuous ∈ [0,1]                    |

### Markov Property Justification

The next state depends **only on the current state and chosen action**, not on the full conversation history. The historical context is implicitly captured via the `customer_satisfaction` variable.

---

## 4. Action Space (A)

The chatbot can perform one of the following actions at each timestep:

### Action Types

| Action      | Description                                       |
| ----------- | ------------------------------------------------- |
| recommend_i | Recommend one of the top 50 products (i ∈ [1,50]) |
| faq_j       | Answer one of 20 frequently asked questions       |
| discount_k  | Offer a discount ∈ {5%, 10%, 15%, 20%}            |
| escalate    | Escalate conversation to human agent              |

This yields a **finite but large discrete action space**, consistent with real-world chatbot systems.

---

## 5. Transition Dynamics (P)

State transitions are **stochastic** and depend on:

* Customer sentiment response
* Purchase decision
* Escalation signal

Formally:

[ s' = T(s, a, \epsilon) ]

Where ( \epsilon ) represents environmental randomness.

### Transition Logic

* **Purchase occurs** → satisfaction set to 0.95
* **Escalation occurs** → satisfaction set to 0.80
* Otherwise, satisfaction drifts based on sentiment:

| Sentiment | Satisfaction Change |
| --------- | ------------------- |
| Positive  | +0.05               |
| Neutral   | 0.00                |
| Negative  | −0.05               |

This models realistic customer reactions while maintaining stochasticity.

---

## 6. Reward Function (R)

The reward function encourages outcomes aligned with business objectives.

### Mathematical Definition

[
R(s,a) = w_1 \cdot \Delta satisfaction + w_2 \cdot purchase_indicator + w_3 \cdot escalation_penalty
]

### Weights Used

| Term | Meaning                  | Weight |
| ---- | ------------------------ | ------ |
| w₁   | Satisfaction improvement | 0.5    |
| w₂   | Purchase / conversion    | 0.35   |
| w₃   | Escalation penalty       | −0.15  |

### Rationale

* **Customer satisfaction** is the primary long-term objective
* **Purchases** directly impact revenue
* **Escalations** are penalized to discourage unnecessary handoffs

---

## 7. Discount Factor (γ)

The discount factor controls the importance of future rewards.

[
\gamma = 0.99
]

### Why Discounting is Used

* Encourages **long-term satisfaction** over short-term gains
* Reflects diminishing importance of distant rewards
* Stabilizes value estimation

### Discounted Return

[
G = \sum_{t=0}^{T} \gamma^t R_t
]

This discounted return is used to approximate the value function.

---

## 8. Value Function Approximation

The state-value function is defined as:

[
V(s) = \mathbb{E}[G | s_0 = s]
]

Because the state space is continuous, we use **linear function approximation**.

### Linear Value Function

[
V(s) \approx \theta^T \phi(s)
]

Where:

* ( \phi(s) ) is a feature vector extracted from the state
* ( \theta ) is a learnable parameter vector

### Feature Vector φ(s)

| Feature               | Encoding    |
| --------------------- | ----------- |
| customer_segment      | One-hot (3) |
| product_category      | One-hot (4) |
| customer_satisfaction | Scalar (1)  |

Total feature dimension = **8**

---

## 9. Policy Evaluation Method

A **random policy** is used as a baseline:

[
\pi(a|s) = \frac{1}{|A|}
]

### Monte-Carlo Evaluation

For each episode:

1. Roll out the policy until termination
2. Compute discounted return G
3. Assign G to all visited states
4. Fit linear regression:

[
\min_\theta \sum (G - \theta^T \phi(s))^2
]

This provides an unbiased estimate of the value function.

---

## 10. Metrics Reported

After 1000 episodes, the following metrics are computed:

| Metric             | Description                        |
| ------------------ | ---------------------------------- |
| Average Reward     | Mean discounted return             |
| Purchase Rate      | Fraction of episodes with purchase |
| Escalation Rate    | Fraction of episodes escalated     |
| Value Coefficients | Learned θ parameters               |

All results are saved in `mdp_report.json`.

---

## 11. Implementation Structure

| Component              | Description                   |
| ---------------------- | ----------------------------- |
| RetailMDP              | Main environment class        |
| reset()                | Initialize episode            |
| step(action)           | Execute action and transition |
| compute_state_features | Feature extraction φ(s)       |
| evaluate_random_policy | Policy evaluation             |

---

## 12. Design Choices & Assumptions

* Conversation history embeddings are abstracted into satisfaction dynamics
* Random policy used for unbiased baseline
* Linear value function chosen for interpretability
* Episodes terminate on purchase or escalation

These choices balance **theoretical correctness** and **implementation simplicity**, as expected at assignment level.

---

## 13. Reproducibility

* Fixed random seeds used
* Deterministic environment initialization
* Compatible with Python ≥ 3.9

All experiments are fully reproducible.

---

## 14. Conclusion

This MDP formulation provides a principled reinforcement learning framework for modeling chatbot–customer interactions. It establishes a strong foundation for future extensions such as policy optimization, RLHF, and deep value approximation.


## Files
- mdp.py : End-to-end pipeline
- mdp_report.json : Metrics output

## Execution
```bash
python mdp.py
