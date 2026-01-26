#!/usr/bin/env python3
"""
Create test users for the Pythia Prophecy frontend.
Run from the pythia_prophecy directory: python scripts/create_test_users.py
"""
import sys
from pathlib import Path

# Add parent to path so we can import api modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from datetime import datetime
from api.database import create_user, get_user_by_email, init_database
from api.auth import hash_password
from api.models import UserInDB, SubscriptionTier

# Test users configuration
TEST_USERS = [
    {
        "email": "free@test.com",
        "first_name": "Free",
        "last_name": "User",
        "password": "free123",
        "tier": SubscriptionTier.FREE,
    },
    {
        "email": "basic@test.com",
        "first_name": "Basic",
        "last_name": "User",
        "password": "basic123",
        "tier": SubscriptionTier.BASIC,
    },
    {
        "email": "pro@test.com",
        "first_name": "Pro",
        "last_name": "User",
        "password": "pro123",
        "tier": SubscriptionTier.PRO,
    },
]


def create_test_users():
    """Create test users if they don't exist."""
    init_database()

    print("Creating test users...\n")

    for user_data in TEST_USERS:
        existing = get_user_by_email(user_data["email"])
        if existing:
            print(f"  [EXISTS] {user_data['email']} ({user_data['tier'].value})")
            continue

        now = datetime.utcnow()
        user = UserInDB(
            id=str(uuid.uuid4()),
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            hashed_password=hash_password(user_data["password"]),
            tier=user_data["tier"],
            email_verified=True,  # Pre-verified for testing
            verification_token=None,
            created_at=now,
            updated_at=now,
        )

        create_user(user)
        print(f"  [CREATED] {user_data['email']} ({user_data['tier'].value})")

    print("\n" + "=" * 50)
    print("Test accounts ready to use:")
    print("=" * 50)
    for user_data in TEST_USERS:
        print(f"  Email: {user_data['email']}")
        print(f"  Password: {user_data['password']}")
        print(f"  Tier: {user_data['tier'].value}")
        print()


if __name__ == "__main__":
    create_test_users()
