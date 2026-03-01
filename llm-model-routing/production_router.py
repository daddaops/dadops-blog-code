"""
Code Block 7: Production router with provider fallbacks and circuit breaker.

From: https://dadops.dev/blog/llm-model-routing/

Production-ready router class with:
- Multi-provider fallbacks per tier
- Circuit breaker (opens after 5 consecutive failures)
- Automatic tier escalation when all providers fail
- Request/error tracking and monitoring

No API key required (providers are injected via constructor).
"""

import time
import logging
from collections import defaultdict

logger = logging.getLogger("llm_router")


class ProductionRouter:
    """LLM router with provider fallbacks, circuit breaker, and monitoring."""

    def __init__(self, route_fn, providers: dict):
        self.route_fn = route_fn          # any of our 4 strategies
        self.providers = providers         # {tier: [provider1, provider2, ...]}
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.circuit_open = set()         # providers currently in circuit-open state
        self.stats = {"tier_dist": defaultdict(int), "fallbacks": 0, "total": 0}

    def call_with_fallback(self, query: str, tier: int) -> str:
        """Try each provider for a tier; fall back to next tier on total failure."""
        for provider in self.providers.get(tier, []):
            if provider in self.circuit_open:
                continue
            try:
                start = time.time()
                result = provider.complete(query)
                latency = (time.time() - start) * 1000
                self._record_success(provider, tier, latency)
                return result
            except Exception as e:
                self._record_failure(provider, tier, e)

        # All providers for this tier failed — escalate
        if tier < 3:
            self.stats["fallbacks"] += 1
            logger.warning(f"Tier {tier} exhausted, escalating to Tier {tier+1}")
            return self.call_with_fallback(query, tier + 1)
        raise RuntimeError("All providers failed across all tiers")

    def route(self, query: str) -> str:
        tier = self.route_fn(query)
        self.stats["total"] += 1
        self.stats["tier_dist"][tier] += 1
        return self.call_with_fallback(query, tier)

    def _record_failure(self, provider, tier, error):
        self.error_counts[provider] += 1
        self.request_counts[provider] += 1
        # Open circuit after 5 consecutive failures
        if self.error_counts[provider] >= 5:
            self.circuit_open.add(provider)
            logger.error(f"Circuit OPEN for {provider} after 5 failures")

    def _record_success(self, provider, tier, latency_ms):
        self.error_counts[provider] = 0   # reset consecutive failures
        self.request_counts[provider] += 1
        logger.info(f"tier={tier} provider={provider} latency={latency_ms:.0f}ms")


if __name__ == "__main__":
    # Test with mock providers
    class MockProvider:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail
        def complete(self, query):
            if self.fail:
                raise ConnectionError(f"{self.name} is down")
            return f"[{self.name}] Response to: {query[:40]}..."
        def __repr__(self):
            return self.name
        def __hash__(self):
            return hash(self.name)

    from heuristic_router import heuristic_route

    logging.basicConfig(level=logging.INFO)

    providers = {
        1: [MockProvider("haiku-a"), MockProvider("haiku-b")],
        2: [MockProvider("sonnet-a")],
        3: [MockProvider("opus-a")],
    }

    router = ProductionRouter(heuristic_route, providers)

    # Normal routing
    result = router.route("What time is it?")
    print(f"Result: {result}")
    print(f"Stats: {dict(router.stats['tier_dist'])}")

    # Test fallback — make tier 1 fail
    providers_with_failure = {
        1: [MockProvider("haiku-a", fail=True), MockProvider("haiku-b", fail=True)],
        2: [MockProvider("sonnet-a")],
        3: [MockProvider("opus-a")],
    }
    router2 = ProductionRouter(heuristic_route, providers_with_failure)
    result = router2.route("What time is it?")
    print(f"\nFallback result: {result}")
    print(f"Fallbacks: {router2.stats['fallbacks']}")
