"""Causal DAG / d-separation / Backdoor Criterion demo.

Implements a simple CausalDAG class and demonstrates the three fundamental
graph structures: fork (confounder), chain (mediator), and collider.
"""

class CausalDAG:
    """A directed acyclic graph for causal reasoning."""
    def __init__(self):
        self.edges = {}  # parent -> list of children
        self.nodes = set()

    def add_edge(self, parent, child):
        self.nodes.update([parent, child])
        self.edges.setdefault(parent, []).append(child)

    def parents(self, node):
        return [p for p, children in self.edges.items() if node in children]

    def descendants(self, node):
        desc = set()
        queue = self.edges.get(node, [])
        while queue:
            current = queue.pop(0)
            if current not in desc:
                desc.add(current)
                queue.extend(self.edges.get(current, []))
        return desc

    def is_backdoor(self, treatment, outcome, conditioning_set):
        """Check if conditioning_set satisfies the backdoor criterion."""
        # Rule 1: No node in conditioning set is a descendant of treatment
        treatment_desc = self.descendants(treatment)
        for node in conditioning_set:
            if node in treatment_desc:
                return False, "Conditioning on a descendant of treatment"

        # Rule 2: conditioning_set blocks all backdoor paths
        # (Simplified: check that all parents of treatment are blocked)
        backdoor_nodes = self.parents(treatment)
        for bd in backdoor_nodes:
            if bd not in conditioning_set:
                return False, f"Unblocked backdoor through '{bd}'"
        return True, "Backdoor criterion satisfied"


if __name__ == "__main__":
    # Example 1: Fork (Confounder) -- Stone Size -> Treatment, Stone Size -> Outcome
    dag = CausalDAG()
    dag.add_edge("StoneSize", "Treatment")
    dag.add_edge("StoneSize", "Recovery")
    dag.add_edge("Treatment", "Recovery")

    valid, msg = dag.is_backdoor("Treatment", "Recovery", {"StoneSize"})
    print(f"Fork -- condition on StoneSize: {valid} ({msg})")

    valid, msg = dag.is_backdoor("Treatment", "Recovery", set())
    print(f"Fork -- condition on nothing:   {valid} ({msg})")

    # Example 2: Chain (Mediator) -- Treatment -> Dosage -> Recovery
    dag2 = CausalDAG()
    dag2.add_edge("Treatment", "Dosage")
    dag2.add_edge("Dosage", "Recovery")
    print(f"\nChain -- no confounders, empty set works:")
    valid, msg = dag2.is_backdoor("Treatment", "Recovery", set())
    print(f"  Condition on nothing: {valid} ({msg})")
    print("  (Do NOT condition on Dosage -- it's a mediator, not a confounder)")

    # Example 3: Collider -- Talent -> Hollywood, Looks -> Hollywood
    dag3 = CausalDAG()
    dag3.add_edge("Talent", "Hollywood")
    dag3.add_edge("Looks", "Hollywood")
    print(f"\nCollider -- Talent and Looks are independent:")
    print(f"  But conditioning on Hollywood creates a spurious association!")
    print(f"  (Among Hollywood actors, less talented ones tend to be better looking)")
