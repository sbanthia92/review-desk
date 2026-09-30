"""The agent registry the orchestrator composes a review from.

The orchestrator never imports concrete agents. Whoever wires the pipeline
(T11's ``reviewdesk.registry`` for real runs, tests for fakes) builds an
``AgentRegistry`` holding one ``Agent`` per ``AgentName`` plus the LLM, search
and fetch providers every agent context shares.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from reviewdesk.contracts import Agent, Fetcher, LLMClient, SearchClient


@dataclass
class AgentRegistry:
    """Agents keyed by name (an ``AgentName`` value) plus shared providers.

    ``llm`` is used by the orchestrator itself (classification, planning,
    debate rulings) and handed to every agent through ``ReviewContext``.
    Agents missing from the registry are simply not scheduled.
    """

    llm: LLMClient
    search: SearchClient
    fetcher: Fetcher
    agents: dict[str, Agent] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        agents: Iterable[Agent] | Mapping[str, Agent],
        *,
        llm: LLMClient,
        search: SearchClient,
        fetcher: Fetcher,
    ) -> AgentRegistry:
        """Build a registry from agents (keyed by ``agent.name``) or a mapping."""
        if isinstance(agents, Mapping):
            table = {str(name): agent for name, agent in agents.items()}
        else:
            table = {agent.name: agent for agent in agents}
        return cls(llm=llm, search=search, fetcher=fetcher, agents=table)

    def register(self, agent: Agent, *, name: str | None = None) -> None:
        """Add or replace an agent (keyed by ``name`` or ``agent.name``)."""
        self.agents[name or agent.name] = agent

    def get(self, name: str) -> Agent | None:
        """The agent registered under ``name``, or None."""
        return self.agents.get(str(name))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.agents
