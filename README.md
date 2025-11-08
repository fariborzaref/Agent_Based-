# Agent-Based Model (ABM) of Institutional Norm Adoption — Dr. Fariborz Aref
# Purpose: simulate diffusion of an institutional norm across departments with
#          heterogeneous thresholds and network structure. Export clean metrics.

# Requirements: pip install mesa networkx numpy pandas matplotlib scipy
# Outputs:
#   Agent_Based/out/abm_timeseries.csv
#   Agent_Based/out/abm_agents_final.csv
#   Agent_Based/figs/abm_adoption_over_time.png
#   Agent_Based/figs/abm_degree_vs_threshold.png
#   Agent_Based/figs/abm_network_snapshot.png

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ----------------------------
# Parameters
# ----------------------------
SEED               = 2025
N_AGENTS           = 300
N_DEPARTMENTS      = 6
AVG_DEGREE         = 6
REWIRING_P         = 0.08             # rewiring for small-world structure
TIMESTEPS          = 80
BASELINE_ADOPT_P   = 0.01             # spontaneous adoption floor
SHOCK_T            = 40               # time of external push
SHOCK_MAG          = 0.15             # additive push to adoption probability at SHOCK_T
SOCIAL_WEIGHT      = 1.6              # weight of neighbor influence
NOISE_SD           = 0.05             # random idiosyncratic noise
THRESH_MEAN        = 0.45
THRESH_SD          = 0.12
DEPT_ASSORT        = 0.35             # probability boost for within-department ties

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Agent
# ----------------------------
class StaffAgent(Agent):
    def __init__(self, unique_id, model, dept, threshold, init_adopt):
        super().__init__(unique_id, model)
        self.dept = dept
        self.threshold = np.clip(threshold, 0.01, 0.99)
        self.adopt = init_adopt  # 0 or 1

    def step(self):
        # neighborhood adoption share
        neighbors = list(self.model.G.neighbors(self.unique_id))
        if len(neighbors) == 0:
            neigh_share = 0.0
        else:
            neigh_share = np.mean([self.model.schedule.agents[n].adopt for n in neighbors])

        # external shock component
        shock = SHOCK_MAG if self.model.t == SHOCK_T else 0.0

        # adoption probability via logistic transform of gap from threshold
        x = SOCIAL_WEIGHT * (neigh_share - self.threshold) + shock
        p_social = 1.0 / (1.0 + np.exp(-x))

        # baseline innovation and idiosyncratic noise
        p = np.clip(BASELINE_ADOPT_P + p_social + np.random.normal(0, NOISE_SD), 0.0, 1.0)

        # once adopted, stay adopted
        if self.adopt == 0 and random.random() < p:
            self.adopt = 1

# ----------------------------
# Model
# ----------------------------
class InstitutionModel(Model):
    def __init__(self):
        super().__init__()
        self.t = 0
        self.schedule = RandomActivation(self)
        self._build_network()
        self._init_agents()
        self.datacollector = DataCollector(
            model_reporters={
                "adoption_rate": lambda m: np.mean([a.adopt for a in m.schedule.agents]),
                "dept_assortativity": self.dept_assortativity,
                "threshold_mean": lambda m: float(np.mean([a.threshold for a in m.schedule.agents])),
                "threshold_sd": lambda m: float(np.std([a.threshold for a in m.schedule.agents])),
                "spearman_deg_threshold": self.spearman_degree_threshold
            }
        )

    def _build_network(self):
        # small-world backbone
        G = nx.watts_strogatz_graph(n=N_AGENTS, k=AVG_DEGREE, p=REWIRING_P, seed=SEED)

        # assign departments
        depts = np.repeat(np.arange(N_DEPARTMENTS), int(np.ceil(N_AGENTS / N_DEPARTMENTS)))[:N_AGENTS]
        np.random.shuffle(depts)
        nx.set_node_attributes(G, {i: int(depts[i]) for i in G.nodes}, "dept")

        # add some department assortative edges
        nodes_by_dept = {d: [n for n, dval in nx.get_node_attributes(G, "dept").items() if dval == d]
                         for d in range(N_DEPARTMENTS)}
        add_edges = int(DEPT_ASSORT * N_AGENTS)
        for _ in range(add_edges):
            d = np.random.randint(0, N_DEPARTMENTS)
            if len(nodes_by_dept[d]) > 2:
                u, v = np.random.choice(nodes_by_dept[d], 2, replace=False)
                if u != v:
                    G.add_edge(int(u), int(v))

        self.G = G

    def _init_agents(self):
        # heterogeneous thresholds
        thresholds = np.random.normal(THRESH_MEAN, THRESH_SD, N_AGENTS)
        # sparse initial adoption seeds
        initial_adopters = set(np.random.choice(range(N_AGENTS), size=max(1, N_AGENTS // 30), replace=False))

        for i in range(N_AGENTS):
            dept = self.G.nodes[i]["dept"]
            agent = StaffAgent(
                unique_id=i,
                model=self,
                dept=dept,
                threshold=thresholds[i],
                init_adopt=1 if i in initial_adopters else 0
            )
            self.schedule.add(agent)

    def dept_assortativity(self):
        # assortativity by department attribute
        return float(nx.attribute_assortativity_coefficient(self.G, "dept"))

    def spearman_degree_threshold(self):
        deg = dict(self.G.degree())
        th = {a.unique_id: a.threshold for a in self.schedule.agents}
        d = []
        t = []
        for k in deg:
            d.append(deg[k])
            t.append(th[k])
        rho, _ = spearmanr(d, t)
        return float(0.0 if np.isnan(rho) else rho)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.t += 1

# ----------------------------
# Run
# ----------------------------
def run_model():
    model = InstitutionModel()
    for _ in range(TIMESTEPS):
        model.step()

    # outputs
    os.makedirs("Agent_Based/out", exist_ok=True)
    os.makedirs("Agent_Based/figs", exist_ok=True)

    # timeseries
    ts = pd.DataFrame(model.datacollector.get_model_vars_dataframe()).reset_index().rename(columns={"index": "t"})
    ts.to_csv("Agent_Based/out/abm_timeseries.csv", index=False)

    # final agent table
    agents = pd.DataFrame({
        "id": [a.unique_id for a in model.schedule.agents],
        "dept": [a.dept for a in model.schedule.agents],
        "threshold": [a.threshold for a in model.schedule.agents],
        "adopt": [a.adopt for a in model.schedule.agents],
        "degree": [model.G.degree[a.unique_id] for a in model.schedule.agents]
    })
    agents.to_csv("Agent_Based/out/abm_agents_final.csv", index=False)

    # figures: restrained typography and sizing
    plt.figure()
    plt.plot(ts["t"], ts["adoption_rate"])
    plt.xlabel("Time")
    plt.ylabel("Adoption rate")
    plt.title("Norm adoption over time")
    plt.savefig("Agent_Based/figs/abm_adoption_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(agents["degree"], agents["threshold"], s=12)
    plt.xlabel("Network degree")
    plt.ylabel("Adoption threshold")
    plt.title("Degree vs threshold")
    plt.savefig("Agent_Based/figs/abm_degree_vs_threshold.png", dpi=300, bbox_inches="tight")
    plt.close()

    # network snapshot: label only a few salient nodes to keep it clean
    pos = nx.spring_layout(model.G, seed=SEED)
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_edges(model.G, pos, alpha=0.15, width=0.5)
    node_sizes = 50 + 40 * np.array([model.G.degree[n] for n in model.G.nodes()])
    nx.draw_networkx_nodes(model.G, pos, node_size=node_sizes)
    # label a small set of high-degree nodes
    high_deg = sorted(model.G.degree, key=lambda x: x[1], reverse=True)[:12]
    labels = {n: str(n) for n, _ in high_deg}
    nx.draw_networkx_labels(model.G, pos, labels=labels, font_size=7)
    plt.title("Network snapshot")
    plt.axis("off")
    plt.savefig("Agent_Based/figs/abm_network_snapshot.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run_model()
