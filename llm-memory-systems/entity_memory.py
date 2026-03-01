"""
Code Block 4: EntityMemory — structured facts about entities with conflict resolution.

From: https://dadops.dev/blog/llm-memory-systems/

Extracts entity-attribute-value triples from conversation messages.
When contradictions are detected (e.g., database changed from PostgreSQL
to MySQL), resolves by recency and logs the change.

No API key required (uses simulated entity extraction).
"""


class EntityMemory:
    """Extract and maintain structured facts about entities."""

    def __init__(self):
        self.entities = {}     # {entity_name: {attribute: value}}
        self.changelog = []    # track what changed and when

    def extract_and_store(self, message, turn_number):
        """Extract entity facts from a message and update the store."""
        # In production: use an LLM with a structured extraction prompt
        triples = self._extract_triples(message)

        for entity, attribute, value in triples:
            if entity not in self.entities:
                self.entities[entity] = {}

            old_value = self.entities[entity].get(attribute)
            if old_value and old_value != value:
                # Contradiction detected — resolve by recency
                self.changelog.append({
                    "turn": turn_number,
                    "entity": entity,
                    "attribute": attribute,
                    "old": old_value,
                    "new": value
                })

            self.entities[entity][attribute] = value

    def _extract_triples(self, message):
        """Simulate entity extraction (in production, use an LLM)."""
        triples = []
        text = message.lower()
        # Simple pattern matching — real systems use LLM extraction
        db_keywords = {"postgresql": "PostgreSQL", "mysql": "MySQL",
                       "mongodb": "MongoDB", "sqlite": "SQLite"}
        for kw, name in db_keywords.items():
            if kw in text:
                triples.append(("Project", "database", name))

        framework_kw = {"react": "React", "vue": "Vue", "angular": "Angular",
                        "express": "Express", "fastapi": "FastAPI", "django": "Django"}
        for kw, name in framework_kw.items():
            if kw in text:
                triples.append(("Project", "framework", name))

        if "jwt" in text:
            triples.append(("API", "auth_method", "JWT"))
        if "oauth" in text:
            triples.append(("API", "auth_method", "OAuth"))

        return triples

    def get_relevant(self, query=""):
        """Get entity facts relevant to the current conversation."""
        lines = []
        for entity, attrs in self.entities.items():
            facts = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            lines.append(f"[{entity}] {facts}")
        return "\n".join(lines)

    def get_changelog(self):
        return self.changelog


if __name__ == "__main__":
    mem = EntityMemory()
    mem.extract_and_store("We're building with React and PostgreSQL", turn_number=1)
    mem.extract_and_store("Authentication will use JWT tokens", turn_number=5)
    mem.extract_and_store("Actually, let's switch to MySQL instead", turn_number=12)

    print(mem.get_relevant())
    # [Project] database: MySQL, framework: React
    # [API] auth_method: JWT

    print(mem.get_changelog())
    # [{'turn': 12, 'entity': 'Project', 'attribute': 'database',
    #   'old': 'PostgreSQL', 'new': 'MySQL'}]
