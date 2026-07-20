"""
tests/conftest.py — Shared pytest fixtures for LEX-DISCOVERY tests.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.graph.state import CaseLawResult, ClientData, ClientFilesState, CaseLawState, DiscoveryState


# ─────────────────────────────────────────────────────────────────────────────
# State fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_client_data() -> ClientData:
    return {
        "metadata": {
            "tenant": "John Mwendwa Doe",
            "landlord": "Kiambu Realty Holdings Ltd",
            "property_address": "Plot 45, Kiambu Road, Nairobi",
            "lease_start": "2022-03-01",
            "lease_end": "2023-02-28",
            "monthly_rent": "KES 45,000",
        },
        "timeline": [
            {"event": "Lease signed", "date": "2022-03-01"},
            {"event": "Written eviction notice issued (18 days)", "date": "2023-01-05"},
            {"event": "Tenant disputes notice", "date": "2023-01-20"},
        ],
        "clauses": [
            "Section 4.2: 30-day written notice required for termination."
        ],
    }


@pytest.fixture
def sample_case_law_results() -> list:
    return [
        {
            "title": "Kamau v. Kiambu County Housing Board",
            "citation": "2021 KLR 456",
            "summary": "18-day notice was insufficient; 30 days required.",
            "relevance_score": 0.95,
        },
        {
            "title": "Mwangi v. Nairobi Realty Corp",
            "citation": "2019 KLR 789",
            "summary": "Notice period runs from date of actual delivery.",
            "relevance_score": 0.88,
        },
    ]


@pytest.fixture
def sample_compliance_gaps() -> list:
    return [
        "Landlord issued only 18 days' notice, violating Section 4.2 (30 days required).",
        "Verbal eviction request on 2022-12-15 does not satisfy written notice requirement.",
    ]


@pytest.fixture
def initial_discovery_state() -> DiscoveryState:
    return {
        "messages": [HumanMessage(content="Tenant disputes 18-day eviction notice.")],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
        "file_path": "tests/fixtures/sample_lease.txt",
    }


@pytest.fixture
def client_files_state(tmp_path) -> ClientFilesState:
    # Create a sample lease text file for testing
    lease_file = tmp_path / "sample_lease.txt"
    lease_file.write_text(
        "LEASE AGREEMENT\n"
        "Tenant: John Doe\nLandlord: Kiambu Realty\n"
        "Section 4.2: 30-day written notice required.\n"
        "2023-01-05: Eviction notice issued (18 days).\n"
    )
    return {
        "file_path": str(lease_file),
        "client_data": None,
        "messages": [],
    }


@pytest.fixture
def case_law_state() -> CaseLawState:
    return {
        "query": "insufficient eviction notice period Kenya tenancy",
        "results": [],
        "messages": [],
    }
