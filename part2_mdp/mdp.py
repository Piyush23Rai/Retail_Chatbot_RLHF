# PART 2: MARKOV DECISION PROCESS (MDP) IMPLEMENTATION
# -------------------------------------------------
# Retail Chatbot – Customer Satisfaction MDP

import random
import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression


class RetailMDP:
    """
    Retail Chatbot MDP Environment
    State:
        - customer_segment: {new, regular, vip}
        - product_category: {Electronics, Clothing, Home, Sports}
        - customer_satisfaction: float in [0,1]
    Actions:
        - recommend_product (50 variants)
        - answer_faq (20 variants)
        - offer_discount {5,10,15,20}
        - escalate_to_human
    """

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # --- State spaces ---
        self.customer_segments = ["new", "regular", "vip"]
        self.product_categories = ["Electronics", "Clothing", "Home", "Sports"]

        # --- Action space ---
        self.actions = (
            [f"recommend_{i}" for i in range(1, 51)] +
            [f"faq_{j}" for j in range(1, 21)] +
            ["discount_5", "discount_10", "discount_15", "discount_20"] +
            ["escalate"]
        )

        # Reward weights
        self.w1 = 0.5   # satisfaction gain
        self.w2 = 0.35  # purchase reward
        self.w3 = -0.15 # escalation penalty

        self.gamma = 0.99
        self.reset()

    # -------------------------------------------------
    # Environment core methods
    # -------------------------------------------------

    def reset(self):
        self.state = {
            "customer_segment": random.choice(self.customer_segments),
            "product_category": random.choice(self.product_categories),
            "customer_satisfaction": round(random.uniform(0.3, 0.7), 2)
        }
        self.done = False
        return self.state

    def step(self, action):
        assert action in self.actions, "Invalid action"

        prev_satisfaction = self.state["customer_satisfaction"]

        # --- Transition dynamics (stochastic) ---
        sentiment = random.choices(
            ["positive", "neutral", "negative"],
            weights=[0.4, 0.4, 0.2]
        )[0]

        purchase = False
        escalation = False

        if action.startswith("recommend"):
            purchase = random.random() < 0.25
        elif action.startswith("discount"):
            purchase = random.random() < 0.35
        elif action == "escalate":
            escalation = True

        # Satisfaction update
        if purchase:
            new_satisfaction = 0.95
        elif escalation:
            new_satisfaction = 0.80
        else:
            delta = {
                "positive": 0.05,
                "neutral": 0.00,
                "negative": -0.05
            }[sentiment]
            new_satisfaction = np.clip(prev_satisfaction + delta, 0, 1)

        self.state["customer_satisfaction"] = round(float(new_satisfaction), 2)

        # --- Reward function ---
        satisfaction_gain = new_satisfaction - prev_satisfaction
        reward = (
            self.w1 * satisfaction_gain +
            self.w2 * int(purchase) +
            self.w3 * int(escalation)
        )

        # Episode termination condition
        if purchase or escalation:
            self.done = True

        info = {
            "purchase": purchase,
            "escalation": escalation,
            "sentiment": sentiment
        }

        return self.state, reward, self.done, info

    # -------------------------------------------------
    # Feature extraction for value function
    # -------------------------------------------------

    def compute_state_features(self, state):
        """phi(s): numeric feature vector"""
        features = []

        # One-hot: customer segment
        for seg in self.customer_segments:
            features.append(1 if state["customer_segment"] == seg else 0)

        # One-hot: product category
        for cat in self.product_categories:
            features.append(1 if state["product_category"] == cat else 0)

        # Satisfaction scalar
        features.append(state["customer_satisfaction"])

        return np.array(features)

    # -------------------------------------------------
    # Policy evaluation
    # -------------------------------------------------

    def evaluate_random_policy(self, episodes=1000):
        all_features = []
        all_returns = []

        total_rewards = []
        purchases = 0
        escalations = 0
        gamma_t = 1.0

        for _ in range(episodes):
            state = self.reset()
            done = False
            episode_reward = 0
            episode_states = []

            while not done:
                action = random.choice(self.actions)
                next_state, reward, done, info = self.step(action)

                episode_states.append(self.compute_state_features(state))
                episode_reward += gamma_t * reward
                gamma_t *= self.gamma

                purchases += int(info["purchase"])
                escalations += int(info["escalation"])

                state = next_state

            # Monte Carlo return assignment
            for phi_s in episode_states:
                all_features.append(phi_s)
                all_returns.append(episode_reward)

            total_rewards.append(episode_reward)

        # Linear value function
        X = np.array(all_features)
        y = np.array(all_returns)

        value_model = LinearRegression()
        value_model.fit(X, y)

        return {
            "average_reward": float(np.mean(total_rewards)),
            "purchase_rate": purchases / episodes,
            "escalation_rate": escalations / episodes,
            "value_function_coefficients": value_model.coef_.tolist(),
            "value_function_intercept": float(value_model.intercept_)
        }



# -------------------------------------------------
# Run evaluation & save JSON report
# -------------------------------------------------

if __name__ == "__main__":
    env = RetailMDP()
    report = env.evaluate_random_policy(episodes=1000)

    with open("part2_mdp/mdp_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("MDP evaluation completed. Report saved to mdp_report.json")
