import pandas as pd
import os

def generate_data(data_dir="../test_data"):
    os.makedirs(data_dir, exist_ok=True)

    # --- Person nodes ---
    person_data = {
        "person_id": [0, 1, 2, 3],
        "budget_per_task": [5000, 7000, 3000, 6000]
    }
    person_df = pd.DataFrame(person_data)
    person_df.to_csv(f"{data_dir}/person_nodes.csv", index=False)

    # --- AI Agent nodes ---
    agent_data = {
        "agent_id": [0, 1, 2, 3],
        "cost_per_task": [4500, 6500, 5500, 4800]
    }
    agent_df = pd.DataFrame(agent_data)
    agent_df.to_csv(f"{data_dir}/agent_nodes.csv", index=False)

    # --- Skill nodes with labels ---
    skill_data = {
        "skill_id": [0, 1, 2, 3, 4],
        "skill_label": [
            "Accounting",
            "Marketing",
            "Software Dev",
            "Data Science",
            "Finance"
        ]
    }
    skill_df = pd.DataFrame(skill_data)
    skill_df.to_csv(f"{data_dir}/skill_nodes.csv", index=False)

    # ------------------------------------------------------------------------
    # Person-Skill edges (toy example)
    # ------------------------------------------------------------------------
    # Person 0 -> Skills {0,4}
    # Person 1 -> Skill  {2}
    # Person 2 -> Skill  {1}
    # Person 3 -> Skill  {3}
    person_skill_edges = {
        "person_id": [0, 1, 2, 3, 0],  # 5 edges total
        "skill_id":  [0, 2, 1, 3, 4]
    }
    person_skill_edges_df = pd.DataFrame(person_skill_edges)
    person_skill_edges_df.to_csv(f"{data_dir}/person_skill_edges.csv", index=False)

    # ------------------------------------------------------------------------
    # AI Agent-Skill edges (toy example)
    # ------------------------------------------------------------------------
    # Agent 0 -> Skill {0}
    # Agent 1 -> Skill {2}
    # Agent 2 -> Skill {3}
    # Agent 3 -> Skills {1,4}
    agent_skill_edges = {
        "agent_id": [0, 1, 2, 3, 3],  # 5 edges total
        "skill_id": [0, 2, 3, 1, 4]
    }
    agent_skill_edges_df = pd.DataFrame(agent_skill_edges)
    agent_skill_edges_df.to_csv(f"{data_dir}/agent_skill_edges.csv", index=False)

    # ------------------------------------------------------------------------
    # Create link prediction edges & labels dynamically
    # ------------------------------------------------------------------------
    # For each (Person, Agent), link_score=1 if they share at least one Skill, else 0
    #
    # We'll systematically gather the Person->Skill sets and Agent->Skill sets,
    # then create all (Person,Agent) pairs with the label determined by intersection.
    #
    # Person->Skill sets (from above):
    #   p0 -> {0,4}
    #   p1 -> {2}
    #   p2 -> {1}
    #   p3 -> {3}
    #
    # Agent->Skill sets:
    #   a0 -> {0}
    #   a1 -> {2}
    #   a2 -> {3}
    #   a3 -> {1,4}

    # Build dictionaries of skill sets for Person & Agent:
    person_skills = {}
    for p, s in zip(person_skill_edges["person_id"], person_skill_edges["skill_id"]):
        person_skills.setdefault(p, set()).add(s)
    # Fill in missing persons with empty sets if needed
    for p in range(4):
        person_skills.setdefault(p, set())

    agent_skills = {}
    for a, s in zip(agent_skill_edges["agent_id"], agent_skill_edges["skill_id"]):
        agent_skills.setdefault(a, set()).add(s)
    # Fill in missing agents with empty sets if needed
    for a in range(4):
        agent_skills.setdefault(a, set())

    # For all 4 x 4 = 16 (Person,Agent) pairs, compute label
    all_person_ids = []
    all_agent_ids = []
    all_scores = []

    for p in range(4):
        for a in range(4):
            # Link score = 1 if intersection of skill sets is non-empty
            shared_skills = person_skills[p].intersection(agent_skills[a])
            score = 1 if len(shared_skills) > 0 else 0
            all_person_ids.append(p)
            all_agent_ids.append(a)
            all_scores.append(score)

    link_prediction_edges = {
        "person_id": all_person_ids,
        "agent_id":  all_agent_ids,
        "link_score": all_scores
    }
    link_prediction_edges_df = pd.DataFrame(link_prediction_edges)
    link_prediction_edges_df.to_csv(f"{data_dir}/link_prediction_edges.csv", index=False)

    print(f"CSV files created in '{data_dir}' with link_score=1 iff Person & Agent share a skill.")

if __name__ == "__main__":
    generate_data()
